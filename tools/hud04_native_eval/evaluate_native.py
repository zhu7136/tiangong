"""PREPARED, NOT ISAACLAB-RUNTIME-VALIDATED: bounded native PhysX evaluation.

No job submission or training API exists in this script. Run only on an approved
existing IsaacLab2.1/IsaacSim4.5 GPU environment. Explicit opt-in is required.
The MuJoCo diagnostic is not the native evaluation result.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True,help='Existing HU_D04 training code checkout')
    p.add_argument('--bundle',type=Path,required=True,help='Unzipped CPU evaluation bundle')
    p.add_argument('--usd',type=Path,required=True,help='Verified training HU_D04_01.usd, with dependent assets')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-gpu-eval',action='store_true')
    p.add_argument('--final-checkpoint',type=Path,required=True)
    p.add_argument('--final-iteration',type=int,default=17309)
    p.add_argument('--hip-knee-limit',type=float,choices=[5.,8.],default=5.)
    p.add_argument('--process-exit-after-result',action='store_true',help='Dedicated child only; bypass known hanging simulator teardown after durable report')
    args=p.parse_args()
    if not args.allow_gpu_eval:p.error('Explicit --allow-gpu-eval is required; no simulator was started.')
    args.repo=args.repo.resolve();args.bundle=args.bundle.resolve();args.usd=args.usd.resolve();args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError('Refusing to overwrite an existing evaluation directory')
    assert args.usd.is_file()
    contract=json.loads((args.bundle/'deployment_contract.json').read_text())
    assert hashlib.sha256(args.final_checkpoint.read_bytes()).hexdigest()==contract['checkpoint_sha256']
    sys.path[:0]=[str(args.repo),str(args.repo/'rsl_rl')];os.chdir(args.repo)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,fast_shutdown=False).app
    started=time.monotonic()
    try:
        import numpy as np
        import torch
        
        from legged_lab.envs import task_registry
        assert torch.cuda.is_available(),'Native GPU runtime required; no CPU fallback'
        cfg,_=task_registry.get_cfgs('hu_d04_run')
        cfg.scene.robot.actuators['legs'].velocity_limit_sim={n:args.hip_knee_limit for n in cfg.scene.robot.actuators['legs'].velocity_limit_sim}
        assert cfg.sim.dt==.005 and cfg.sim.decimation==4
        assert cfg.robot.action_scale==.25 and cfg.robot.actor_obs_history_length==10
        assert cfg.gait.gait_cycle==.5
        assert [cfg.gait.gait_phase_offset_l,cfg.gait.gait_phase_offset_r]==[.6,.1]
        assert [cfg.gait.gait_air_ratio_l,cfg.gait.gait_air_ratio_r]==[.6,.6]
        cfg.scene.num_envs=18;cfg.scene.env_spacing=10.;cfg.scene.seed=42
        cfg.scene.terrain_type='plane';cfg.scene.terrain_generator=None
        cfg.scene.max_episode_length_s=20.
        cfg.scene.robot.spawn.usd_path=str(args.usd)
        cfg.noise.add_noise=False;cfg.domain_rand.action_delay.enable=False
        # A deterministic, nominal-parameter comparison, not a robustness test.
        for name in ['add_base_mass','reset_base','reset_robot_joints','push_robot']:
            setattr(cfg.domain_rand.events,name,None)
        material=cfg.domain_rand.events.physics_material
        material.params.update(static_friction_range=(.8,.8),dynamic_friction_range=(.8,.8),restitution_range=(0.,0.))
        cfg.commands.debug_vis=False
        cfg.commands.rel_standing_envs=0.;cfg.commands.rel_heading_envs=0.
        cfg.scene.height_scanner.enable_height_scan=False
        cfg.scene.lidar.enable_lidar=False;cfg.scene.depth_camera.enable_depth_camera=False
        env=task_registry.get_task_class('hu_d04_run')(cfg,True)
        assert env.num_actions==31
        assert list(env.robot.joint_names)==contract['joint_names_in_policy_order']
        assert torch.allclose(env.robot.data.default_joint_pos[0],torch.tensor(contract['default_q'],device=env.device),atol=1e-6)
        expected_limits=json.loads((args.bundle/'native_eval_plan.json').read_text())['joint_velocity_limits_rad_s']
        for n in cfg.scene.robot.actuators['legs'].velocity_limit_sim:expected_limits[n]=args.hip_knee_limit
        actual_limits=env.robot.data.joint_vel_limits[0].detach().cpu().numpy()
        assert np.allclose(actual_limits,[expected_limits[n] for n in env.robot.joint_names],atol=1e-5),'Actual PhysX joint limits differ'
        print(json.dumps({'stage':'hud04_limit8_eval_limits_verified','hip_knee_limit':args.hip_knee_limit,'actual_limits':actual_limits.tolist()}),flush=True)
        metadata=[{'iteration':it,'command_vx':v,'seed':s} for it in [15500,args.final_iteration] for v in [1.2,2.,2.8] for s in range(3)]
        command=torch.tensor([r['command_vx'] for r in metadata],device=env.device)
        # Use the native command object and observation builder throughout.
        def fixed_commands(dt=0.):
            q=env.robot.data.root_quat_w;w,x,y,z=q.unbind(-1)
            yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            env.command_generator.command[:,0]=command
            env.command_generator.command[:,1]=0.
            env.command_generator.command[:,2]=torch.clamp(-.5*yaw,-1.57,1.57)
        env.command_generator.compute=fixed_commands
        blocked_reset_calls=[]
        def no_auto_reset(ids):
            if len(ids):blocked_reset_calls.append({'step':int(env.episode_length_buf.max().item()),'ids':ids.detach().cpu().tolist()})
        env.reset=no_auto_reset
        q=env.robot.data.default_joint_pos.clone();dq=torch.zeros_like(q)
        for i,row in enumerate(metadata):
            if row['seed']:
                q[i]+=torch.tensor(np.random.default_rng(row['seed']).normal(0,.002,31),device=env.device,dtype=q.dtype)
        root_state=env.robot.data.default_root_state.clone()
        root_state[:,:3]+=env.scene.env_origins
        root_state[:,7:]=0.
        env.robot.write_root_state_to_sim(root_state)
        env.robot.write_joint_state_to_sim(q,dq)
        env.scene.write_data_to_sim();env.sim.forward();fixed_commands()
        env.actor_obs_buffer.reset();env.critic_obs_buffer.reset();env.action_buffer.reset()
        env.episode_length_buf[:]=0;env.gait_phase[:]=0
        obs,_=env.compute_observations();assert obs.shape==(18,1080)
        def load_actor(path,iteration,expected_sha):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['iter']==iteration
            assert 'obs_norm_state_dict' not in checkpoint
            state={k.removeprefix('actor.'):v for k,v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
            layers=[]
            weights=sorted((k for k in state if k.endswith('.weight')),key=lambda k:int(k.split('.')[0]))
            for j,k in enumerate(weights):
                w=state[k];layers.append(torch.nn.Linear(w.shape[1],w.shape[0]))
                if j<len(weights)-1:layers.append(torch.nn.ELU())
            actor=torch.nn.Sequential(*layers).eval();actor.load_state_dict(state,strict=True)
            assert actor[0].in_features==1080 and actor[-1].out_features==31
            return actor
        torch.set_num_threads(1)
        sessions={15500:load_actor(args.repo/'resume-inputs/model_15500.pt',15500,'f3d91b4b0e536c9151d5d27c6a6adc30e44e968010547f3ffec359f16af856c7'),args.final_iteration:load_actor(args.final_checkpoint,args.final_iteration,contract['checkpoint_sha256'])}
        print(json.dumps({'stage':'hud04_native_eval_actors_loaded','training_updates':0,'iterations':[15500,args.final_iteration]}),flush=True)
        args.output.mkdir(parents=True,exist_ok=False)
        active=np.ones(18,dtype=bool);traces=[[] for _ in range(18)];failures=[None]*18
        # 600 control ticks = 12 simulated seconds, 2s warmup + 10s measurement.
        with torch.inference_mode(),(args.output/'frames.jsonl').open('w') as log:
            for step in range(600):
                if time.monotonic()-started>240:raise TimeoutError('240-second in-process wall budget reached')
                x=obs.detach().cpu().numpy();actions=np.zeros((18,31),dtype=np.float32)
                for i,row in enumerate(metadata):
                    if active[i]:actions[i]=sessions[row['iteration']](torch.from_numpy(x[i:i+1])).numpy()[0]
                obs,_,done,_=env.step(torch.from_numpy(actions).to(env.device))
                pos=(env.robot.data.root_pos_w-env.scene.env_origins).detach().cpu().numpy()
                vel=env.robot.data.root_lin_vel_w.detach().cpu().numpy()
                gravity=env.robot.data.projected_gravity_b.detach().cpu().numpy()
                qs=env.robot.data.root_quat_w.detach().cpu().numpy()
                terminated=done.detach().cpu().numpy();t=(step+1)*.02
                joint_dq=env.robot.data.joint_vel.detach().cpu().numpy()
                torque=env.robot.data.applied_torque.detach().cpu().numpy()
                effort_limits=env.robot.data.joint_effort_limits[0].detach().cpu().numpy()
                for i in np.flatnonzero(active):
                    w,xq,yq,zq=qs[i];yaw=float(np.degrees(np.arctan2(2*(w*zq+xq*yq),1-2*(yq*yq+zq*zq))))
                    tilt=float(np.degrees(np.arccos(np.clip(-gravity[i,2],-1,1))))
                    row={'step':step+1,'env':int(i),'t':t,'x':float(pos[i,0]),'y':float(pos[i,1]),'z':float(pos[i,2]),'vx':float(vel[i,0]),'vy':float(vel[i,1]),'yaw_deg':yaw,'tilt_deg':tilt}
                    row.update(joint_vel_rad_s=joint_dq[i].tolist(),applied_torque_nm=torque[i].tolist())
                    traces[i].append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                    reason=('native_termination_contact' if terminated[i] else 'base_height_below_0.55m' if pos[i,2]<.55 else 'tilt_above_60deg' if tilt>60 else None)
                    if reason:failures[i]=reason;active[i]=False
                if (step+1)%50==0:log.flush();print(json.dumps({'step':step+1,'still_active':int(active.sum())}),flush=True)
                if not active.any():break
        results=[]
        for i,meta in enumerate(metadata):
            window=[r for r in traces[i] if r['t']>2.]
            complete=bool(active[i] and len(traces[i])==600)
            measured=float(np.mean([r['vx'] for r in window])) if complete else None
            mae_y=float(np.mean([abs(r['vy']) for r in window])) if complete else None
            mae_heading=float(np.mean([abs(r['yaw_deg']) for r in window])) if complete else None
            qualifies=bool(complete and abs(measured-meta['command_vx'])<=.1*meta['command_vx'] and mae_y<.3 and mae_heading<15.)
            results.append({**meta,'survived_12s':complete,'qualified_target_tracking':qualifies,'failure':failures[i],'elapsed_s':len(traces[i])*.02,'qualified_window_vx_mps':measured,'mean_abs_vy':mae_y,'mean_abs_heading_deg':mae_heading})
        report={'classification':'native_physx_evaluation','training_updates':0,'wall_seconds':time.monotonic()-started,'auto_reset_disabled':True,'blocked_reset_calls':blocked_reset_calls,'inference_backend':'torch_cpu_deterministic_actor','checkpoint_sha256':contract['checkpoint_sha256'],'results':results,'joint_names':list(env.robot.joint_names),'joint_velocity_limits':actual_limits.tolist(),'joint_effort_limits':effort_limits.tolist(),'hip_knee_limit':args.hip_knee_limit,'cleanup_mode':'dedicated_process_exit' if args.process_exit_after_result else 'app.close'}
        (args.output/'native_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({'result_file':str(args.output/'native_results.json'),'completed':True}),flush=True)
        if args.process_exit_after_result:
            # Report and trace files are closed. A dedicated child-process exit
            # releases GPU resources without claiming graceful simulator cleanup.
            sys.stdout.flush();os._exit(0)
    finally:
        app.close(wait_for_replicator=False)
        print(json.dumps({'stage':'hud04_native_eval_cleanup_completed'}),flush=True)

if __name__=='__main__':main()