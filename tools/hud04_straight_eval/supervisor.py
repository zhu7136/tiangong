"""Supervise only this evaluation's child process; no platform mutations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True, help='Verified IsaacLab interpreter')
    p.add_argument('--receipt', required=True)
    p.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    worker_args = args.worker_args
    if worker_args and worker_args[0] == '--':
        worker_args = worker_args[1:]
    if not worker_args:
        raise ValueError('Worker arguments are required')
    receipt = Path(args.receipt)
    if receipt.exists():
        raise FileExistsError('Refusing to overwrite an earlier execution receipt')
    events = queue.Queue(maxsize=256)
    # Default-deny inheritance: arbitrary personal-secret names are not forwarded.
    names = {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
             'LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
             'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR', 'ACCEPT_EULA',
             'PRIVACY_CONSENT', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}
    prefixes = ('CUDA_', 'NVIDIA_', 'ISAAC_', 'OMNI_', 'CARB_', 'KIT_', 'PHYSX_')
    child_env = {k: v for k, v in os.environ.items()
                 if (k in names or k.startswith(prefixes))
                 and not any(tag in k.upper() for tag in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET'))}
    child_env['PYTHONUNBUFFERED'] = '1'
    child = subprocess.Popen([args.python, str(Path(__file__).with_name('straight_eval.py')),
                              *worker_args], env=child_env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def read():
        # Non-event simulator output is deliberately not mirrored or persisted.
        # Bound line memory so abnormal stdout cannot grow the supervisor indefinitely.
        while True:
            line = child.stdout.readline(65537)
            if not line:
                break
            if len(line) > 65536 or not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
                stage = event.get('stage', '')
                if stage.startswith('straight_eval_'):
                    events.put(event, timeout=1)
            except (ValueError, AttributeError, queue.Full):
                continue
        events.put(None)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    started = time.monotonic()
    phase_deadline = started + 300
    timed_out = False
    cleanup = False
    report_verified = False
    last_stage = 'initialization'
    stream_closed = False
    try:
        while not stream_closed:
            now = time.monotonic()
            if now > min(phase_deadline, started + 600):
                timed_out = True
                break
            try:
                event = events.get(timeout=0.25)
            except queue.Empty:
                if child.poll() is not None and not thread.is_alive():
                    break
                continue
            if event is None:
                stream_closed = True
                continue
            last_stage = event['stage']
            if last_stage == 'straight_eval_rollout_started':
                phase_deadline = now + 180
            elif last_stage == 'straight_eval_cleanup_started':
                phase_deadline = min(phase_deadline, now + 60)
            elif last_stage == 'straight_eval_cleanup_completed':
                cleanup = event.get('cleanup_ok') is True
            elif last_stage == 'straight_eval_report_evidence':
                raw = json.dumps(event['report'], sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode()
                report_verified = (len(raw) <= 32768 and len(raw) == event['bytes']
                                   and hashlib.sha256(raw).hexdigest() == event['sha256'])
                if not report_verified:
                    raise ValueError('Child report evidence mismatch')
            print(json.dumps(event, allow_nan=False), flush=True)
        if stream_closed and child.poll() is None:
            try:
                child.wait(timeout=max(0.01, min(phase_deadline, started + 600) - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        result = dict(stage='straight_eval_worker_finished', child_returncode=child.returncode,
                      watchdog_triggered=timed_out, cleanup_completed=cleanup,
                      report_verified=report_verified, last_stage=last_stage,
                      worker_wall_seconds=time.monotonic() - started,
                      note='Worker watchdog is not a platform billing cap.')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x') as f:
            json.dump(result, f, allow_nan=False, indent=2)
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if child.returncode == 0 and not timed_out and cleanup and report_verified else 1


if __name__ == '__main__':
    raise SystemExit(main())