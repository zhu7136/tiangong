"""One bounded GPU/IsaacLab runtime smoke. No training, assets or credentials."""
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def emit(stage, **values):
    print(json.dumps(dict(stage='hud04_gpu_probe_' + stage, **values), allow_nan=False), flush=True)


def runtime_environment(source):
    names = {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
             'LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
             'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR', 'ACCEPT_EULA',
             'PRIVACY_CONSENT', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'EXP_PATH', 'ISAACSIM_PATH'}
    prefixes = ('CUDA_', 'NVIDIA_', 'ISAAC_', 'OMNI_', 'CARB_', 'KIT_', 'PHYSX_')
    result = {k: v for k, v in source.items()
              if (k in names or k.startswith(prefixes))
              and not any(tag in k.upper() for tag in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET'))}
    result['PYTHONUNBUFFERED'] = '1'
    return result


def child():
    import torch
    versions = {}
    for package in ('isaaclab', 'isaacsim', 'isaacsim-core'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'distribution_not_registered'
    available = torch.cuda.is_available()
    emit('runtime', python=sys.version.split()[0], torch=torch.__version__,
         torch_cuda=torch.version.cuda, cuda_available=available, versions=versions)
    if not available:
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    a = torch.ones((32, 32), device=device)
    passed = bool(torch.all(a @ a == 32).item())
    torch.cuda.synchronize()
    emit('cuda', device=torch.cuda.get_device_name(0), tensor_test_passed=passed)
    if not passed:
        raise RuntimeError('CUDA tensor test failed')
    from isaaclab.app import AppLauncher
    emit('app_initializing', exp_path_present=bool(os.environ.get('EXP_PATH')),
         isaacsim_path_present=bool(os.environ.get('ISAACSIM_PATH')))
    app = AppLauncher(headless=True).app
    try:
        import isaaclab.sim as sim_utils
        context = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device='cuda:0'))
        context.reset()
        for _ in range(10):
            context.step(render=False)
        emit('physx', steps=10, physics_dt=0.005, simulated_seconds=0.05,
             context_device=str(context.device), assets_loaded=0, optimizer_updates=0)
    finally:
        emit('cleanup_started')
        app.close()
        emit('cleanup_completed')
    emit('passed', note='Runtime plumbing only; no checkpoint or robot evaluation.')


def main():
    if sys.argv[1:] == ['--child']:
        try:
            child()
            return 0
        except Exception as error:
            detail = {}
            if isinstance(error, KeyError) and error.args and error.args[0] in ('EXP_PATH', 'ISAACSIM_PATH'):
                detail['missing_runtime_variable'] = error.args[0]
            emit('failed', error_type=type(error).__name__, **detail)
            return 1
    # Allowlist runtime variables only; never forward account credentials.
    env = runtime_environment(os.environ)
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, start_new_session=True)
    seen = set()
    def read():
        while True:
            line = proc.stdout.readline(65537)
            if not line:
                return
            if len(line) > 65536 or not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
                if event.get('stage', '').startswith('hud04_gpu_probe_'):
                    seen.add(event['stage'])
                    print(json.dumps(event, allow_nan=False), flush=True)
            except (ValueError, AttributeError):
                continue
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started = time.monotonic()
    timed_out = False
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        reader.join(timeout=2)
    passed = (proc.returncode == 0 and not timed_out
              and 'hud04_gpu_probe_passed' in seen
              and 'hud04_gpu_probe_cleanup_completed' in seen)
    emit('finished', passed=passed, child_returncode=proc.returncode, timed_out=timed_out,
         wall_seconds=time.monotonic() - started)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())