"""Single simulation-only 8 rad/s continuation, bounded at iteration boundaries."""
import argparse,json,os,sys,time
from pathlib import Path

class TrainingBudgetReached(Exception):pass

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1800,choices=[1800])
    args=p.parse_args();root=args.repo.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(root),str(root/'rsl_rl')];os.chdir(root)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,fast_shutdown=False).app
    import torch,numpy as np,random
    from legged_lab.envs import task_registry
    from rsl_rl.runners import AmpOnPolicyRunner
    from isaaclab.utils.io import dump_yaml
    assert torch.cuda.is_available(),'No CPU fallback'
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    cfg,train_cfg=task_registry.get_cfgs('hu_d04_run')
    cfg.scene.num_envs=4096;cfg.scene.seed=train_cfg.seed
    cfg.commands.ranges.lin_vel_x=(1.2,2.8)
    cfg.commands.debug_vis=False
    cfg.scene.robot.spawn.usd_path=str(root/'humanoid-description/HU_D04_description/usd/HU_D04_01.usd')
    legs=cfg.scene.robot.actuators['legs']
    assert len(legs.velocity_limit_sim)==8 and set(legs.velocity_limit_sim.values())=={5.}
    original_effort=dict(legs.effort_limit_sim)
    legs.velocity_limit_sim={name:8. for name in legs.velocity_limit_sim}
    assert legs.effort_limit_sim==original_effort
    train_cfg.logger='tensorboard'
    train_cfg.max_iterations=2000;train_cfg.save_interval=100
    env=task_registry.get_task_class('hu_d04_run')(cfg,True)
    assert env.num_actions==31
    actual=env.robot.data.joint_vel_limits[0]
    for name in legs.velocity_limit_sim:
        assert abs(actual[env.robot.joint_names.index(name)].item()-8.)<1e-5
    assert torch.all(env.robot.data.joint_effort_limits[0][[env.robot.joint_names.index(n) for n in legs.velocity_limit_sim]]==140)
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),train_cfg)
    runner=AmpOnPolicyRunner(env,train_cfg.to_dict(),log_dir=str(out/'training_log'),device=train_cfg.device)
    checkpoint=root/'resume-inputs/model_15500.pt'
    runner.load(str(checkpoint),load_optimizer=True)
    assert runner.current_learning_iteration==15500
    assert runner.alg.optimizer.state
    runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
    assert torch.isfinite(runner.alg.policy.std).all() and (runner.alg.policy.std>0).all()
    # Retain the original PPO implementation/config. Only log additional native
    # telemetry and stop at a completed optimizer boundary; no reward changes.
    original_log=runner.log
    telemetry={};updates=0;began=time.monotonic();last_iter=15500
    def sample_telemetry():
        telemetry['mean_body_vx']=float(env.robot.data.root_lin_vel_b[:,0].mean())
        telemetry['mean_world_vx']=float(env.robot.data.root_lin_vel_w[:,0].mean())
        telemetry['mean_command_x']=float(env.command_generator.command[:,0].mean())
        dq=env.robot.data.joint_vel.abs();tau=env.robot.data.applied_torque.abs()
        telemetry['joint_velocity_near_limit_fraction']=float((dq>=.98*env.robot.data.joint_vel_limits).float().mean())
        telemetry['joint_effort_near_limit_fraction']=float((tau>=.98*env.robot.data.joint_effort_limits).float().mean())
        telemetry['joint_abs_velocity_max']=float(dq.max())
        telemetry['joint_abs_effort_max']=float(tau.max())
        values=getattr(env.reward_manager,'_step_reward',None)
        if isinstance(values,torch.Tensor):
            telemetry['weighted_reward_term_last_step_means']={str(n):float(values[:,i].mean()) for i,n in enumerate(env.reward_manager.active_terms)}
    log_file=(out/'training_metrics.jsonl').open('x')
    def log(locs,*a,**kw):
        nonlocal updates,last_iter
        sample_telemetry()
        original_log(locs,*a,**kw);updates+=1;last_iter=int(locs['it'])
        event={'stage':'hud04_limit8_update','iteration':last_iter,'updates':updates,'optimizer_wall_seconds':time.monotonic()-began,'learning_rate':float(runner.alg.learning_rate),'losses':{k:float(v) for k,v in locs['loss_dict'].items()},'telemetry_sampling':'last_control_step_of_each_rollout',**telemetry}
        log_file.write(json.dumps(event,allow_nan=False)+'\n');log_file.flush()
        for key,value in telemetry.items():
            if isinstance(value,(int,float)):runner.writer.add_scalar('Limit8/'+key,value,last_iter)
        if updates==1 or updates%10==0:print(json.dumps(event,allow_nan=False),flush=True)
        if time.monotonic()-began>=args.seconds-30:raise TrainingBudgetReached()
    runner.log=log
    print(json.dumps({'stage':'hud04_limit8_training_started','num_envs':4096,'seed':train_cfg.seed,'start_iteration':15500,'hip_knee_limit':8.,'actual_limits':actual.detach().cpu().tolist(),'max_optimizer_wall_seconds':1800,'max_updates':2000}),flush=True)
    stop_reason='iteration_cap'
    try:runner.learn(num_learning_iterations=2000,init_at_random_ep_len=True)
    except TrainingBudgetReached:stop_reason='time_budget_at_completed_update'
    finally:log_file.close()
    assert updates>0
    runner.current_learning_iteration=15500+updates;last_iter=runner.current_learning_iteration
    final=out/'final_checkpoint.pt';runner.save(str(final),infos={'environment_yaml':(out/'env.yaml').read_text(),'agent_yaml':(out/'agent.yaml').read_text(),'simulation_only':True,'hip_knee_limit':8.})
    runner.writer.flush();runner.writer.close()
    import hashlib
    report={'stage':'hud04_limit8_training_completed','updates':updates,'final_iteration':last_iter,'final_checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'final_checkpoint_bytes':final.stat().st_size,'optimizer_wall_seconds':time.monotonic()-began,'stop_reason':stop_reason,'hip_knee_limit':8.,'actual_joint_velocity_limits':actual.detach().cpu().tolist(),'last_telemetry':telemetry,'simulator_cleanup':'dedicated_process_exit_after_durable_checkpoint'}
    (out/'training_result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,allow_nan=False),flush=True)
    # Avoid the previously demonstrated IsaacSim teardown hang. This dedicated
    # process owns the simulator; process exit releases it, not graceful close.
    sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()