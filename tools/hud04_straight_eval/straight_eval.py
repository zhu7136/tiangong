"""Bounded inference-only IsaacLab evaluation. Assets are local private inputs.

Run using the selected IsaacSim/IsaacLab interpreter, through supervisor.py.
This file does not download inputs, create tasks, train, or accept credentials.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from eval_core import emit_report, sha256, summarize, validate_drives, verify_inputs


class EpisodeEnded(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--usd', required=True)
    parser.add_argument('--usd-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        raise ValueError('Simulation duration must be in (0, 30] seconds')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = verify_inputs(args.manifest, args.checkpoint, args.policy)
    if len(args.usd_sha256) != 64 or sha256(args.usd) != args.usd_sha256:
        raise ValueError('USD identity mismatch')
    repository = Path(args.repository).resolve()
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != args.expected_commit or len(commit) != 40:
        raise ValueError('Worker source commit mismatch')
    os.chdir(repository)
    sys.path[:0] = [str(repository), str(repository / 'rsl_rl')]
    print(json.dumps({'stage': 'straight_eval_inputs_verified'}), flush=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    cleanup_ok = False
    try:
        import numpy as np
        import torch
        from legged_lab.envs import task_registry

        cfg, _ = task_registry.get_cfgs('hu_d04_run')
        expected_cfg = manifest['env_cfg']
        # Policy-interface drift must fail before physics is interpreted.
        actual_cfg = cfg.to_dict()
        for section in ('robot', 'normalization', 'gait'):
            if json.loads(json.dumps(actual_cfg[section], default=str)) != expected_cfg[section]:
                raise ValueError('Policy configuration differs from runtime manifest: ' + section)
        cfg.device = 'cuda:0'
        cfg.scene.num_envs = 1
        cfg.scene.seed = args.seed
        cfg.scene.max_episode_length_s = args.seconds + 5
        cfg.scene.terrain_generator = None
        cfg.scene.terrain_type = 'plane'
        cfg.scene.robot.spawn.usd_path = str(Path(args.usd).resolve())
        cfg.noise.add_noise = False
        cfg.commands.rel_standing_envs = 0.0
        cfg.commands.rel_heading_envs = 1.0
        cfg.commands.heading_command = True
        cfg.commands.heading_control_stiffness = expected_cfg['commands']['heading_control_stiffness']
        cfg.commands.debug_vis = False
        cfg.commands.resampling_time_range = (60.0, 60.0)
        cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
        cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.ranges.heading = (0.0, 0.0)
        cfg.commands.ranges.ang_vel_z = tuple(expected_cfg['commands']['ranges']['ang_vel_z'])
        cfg.sim.dt = expected_cfg['sim']['dt']
        cfg.sim.decimation = expected_cfg['sim']['decimation']
        cfg.domain_rand.events.push_robot = None
        reset = cfg.domain_rand.events.reset_base.params
        reset['pose_range'] = {k: (0.0, 0.0) for k in reset['pose_range']}
        reset['velocity_range'] = {k: (0.0, 0.0) for k in reset['velocity_range']}
        cfg.domain_rand.events.reset_robot_joints.params.update(
            position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
        env = task_registry.get_task_class('hu_d04_run')(cfg, True)
        if not math.isclose(env.step_dt, 0.02, abs_tol=1e-9):
            raise ValueError('Expected 20ms policy period')
        robot, view = env.robot, env.robot.root_physx_view

        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

        def row(value):
            value = array(value)[0]
            if value.shape != (31,) or not np.isfinite(value).all():
                raise ValueError('Invalid measured joint vector')
            return value.tolist()

        runtime = dict(joint_names=list(robot.joint_names),
                       default_q=row(robot.data.default_joint_pos),
                       default_dq=row(robot.data.default_joint_vel),
                       stiffness=row(view.get_dof_stiffnesses()),
                       damping=row(view.get_dof_dampings()),
                       effort_limit=row(view.get_dof_max_forces()))
        (out / 'measured_drives.json').write_text(json.dumps(runtime, allow_nan=False))
        errors = validate_drives(manifest, runtime)
        routed = np.zeros(31, dtype=bool)
        for group in manifest['env_cfg']['scene']['robot']['actuators'].values():
            for index, name in enumerate(robot.joint_names):
                if any(re.fullmatch(pattern, name) for pattern in group['joint_names_expr']):
                    routed[index] = True
        if int(routed.sum()) != 20:
            raise ValueError('Expected audited 20-joint actuator routing')
        physics = {'body_names': list(robot.body_names), 'startup_randomization': 'retained_seeded',
                   'physics_dt': cfg.sim.dt, 'decimation': cfg.sim.decimation,
                   'self_collisions': cfg.scene.robot.spawn.articulation_props.enabled_self_collisions}
        for key, getter in [('masses', 'get_masses'), ('inertias', 'get_inertias'),
                            ('materials', 'get_material_properties'), ('armature', 'get_dof_armatures')]:
            if hasattr(view, getter):
                physics[key] = array(getattr(view, getter)()).tolist()
            else:
                physics[key] = {'unavailable_getter': getter}
        physics_path = out / 'physics.json'
        physics_path.write_text(json.dumps(physics, allow_nan=False))
        policy = torch.jit.load(args.policy, map_location=env.device).eval()
        env.command_generator.compute(0.0)
        obs, _ = env.get_observations()
        if tuple(obs.shape) != (1, 1080):
            raise ValueError('Expected 1x1080 observation history')

        def sample(t):
            q = array(robot.data.root_quat_w)[0]
            q = q / np.linalg.norm(q)
            w, x, y, z = map(float, q)
            pos = array(robot.data.root_pos_w)[0]
            velocity = array(robot.data.root_lin_vel_b)[0]
            command = array(env.command_generator.command)[0]
            yaw = math.degrees(math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z)))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1-2*(x*x+y*y)))))
            return dict(t=float(t), x=float(pos[0]), y=float(pos[1]), height=float(pos[2]),
                        vx=float(velocity[0]), vy=float(velocity[1]), yaw_deg=yaw,
                        tilt_deg=tilt, command_vx=float(command[0]), command_vy=float(command[1]),
                        yaw_command=float(command[2]))

        trace = [sample(0.0)]
        if abs(trace[0]['yaw_deg']) > 1e-3 or abs(trace[0]['vx']) > 1e-4 or abs(trace[0]['vy']) > 1e-4:
            raise ValueError('Reset did not produce zero yaw and zero horizontal velocity')
        original_reset = env.reset

        def guarded_reset(ids):
            if len(ids):
                raise EpisodeEnded('Environment requested reset; terminal state preserved')
            return original_reset(ids)

        env.reset = guarded_reset
        reason = 'duration_reached'
        steps = round(args.seconds / env.step_dt)
        if not math.isclose(steps * env.step_dt, args.seconds, abs_tol=1e-6):
            raise ValueError('Duration must be a multiple of the policy period')
        print(json.dumps({'stage': 'straight_eval_rollout_started', 'steps': steps}), flush=True)
        for step in range(1, steps + 1):
            with torch.inference_mode():
                action = policy(obs)
                if tuple(action.shape) != (1, 31) or not torch.isfinite(action).all():
                    raise ValueError('Invalid policy action')
                try:
                    obs, _, _, _ = env.step(action)
                except EpisodeEnded:
                    reason = 'environment_reset_requested'
            if step == 1:
                # This private buffer is audited in the supplied articulation source.
                # A missing or changed interface is an incompatibility, not permission to guess.
                targets = array(robot._joint_pos_target_sim)[0]
                proposed = array(action)[0].clip(-env.clip_actions, env.clip_actions) * env.action_scale
                proposed = proposed + np.asarray(runtime['default_q'])
                proposed[~routed] = 0.0
                if not np.allclose(targets, proposed, atol=1e-4, rtol=0):
                    raise ValueError('Effective actuator target routing differs from CPU reference')
            trace.append(sample(step * env.step_dt))
            if trace[-1]['height'] < 0.5 or trace[-1]['tilt_deg'] > 60:
                reason = 'height_or_tilt_limit'
            if reason != 'duration_reached':
                break
            if step % 250 == 0:
                print(json.dumps({'stage': 'straight_eval_progress', 'simulation_seconds': step*env.step_dt}), flush=True)
        # Validate all rows even when no samples remain after warmup.
        metrics = summarize(trace)
        (out / 'trace.json').write_text(json.dumps(trace, allow_nan=False))
        report = dict(schema_version=1, metrics=metrics, stop_reason=reason, source_commit=commit,
                      routed_joint_count=int(routed.sum()),
                      reset_count=0, optimizer_updates=0, seed=args.seed,
                      checkpoint_sha256=manifest['checkpoint_sha256'],
                      policy_sha256=manifest['policy_sha256'], usd_sha256=args.usd_sha256,
                      drive_max_errors=errors, physics_sha256=sha256(physics_path),
                      trace_sha256=sha256(out / 'trace.json'),
                      startup_randomization='retained_seeded_not_nominal_model_equivalence',
                      policy_execution='exact_reference_torchscript_no_reexport',
                      python_version=sys.version.split()[0], torch_version=torch.__version__)
        (out / 'report.json').write_text(json.dumps(report, allow_nan=False, indent=2))
        emit_report(report)
    finally:
        print(json.dumps({'stage': 'straight_eval_cleanup_started'}), flush=True)
        app.close()
        cleanup_ok = True
        print(json.dumps({'stage': 'straight_eval_cleanup_completed', 'cleanup_ok': cleanup_ok}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not mirror exception text: SDK failures may contain private URLs.
        print(json.dumps({'stage': 'straight_eval_failed',
                          'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)