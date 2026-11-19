"""Pure-Python evidence utilities; no simulation or training imports."""
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_inputs(manifest_path, checkpoint, policy):
    manifest = json.loads(Path(manifest_path).read_text())
    for field, path in [('checkpoint_sha256', checkpoint), ('policy_sha256', policy)]:
        expected = manifest.get(field)
        if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
            raise ValueError('Input identity mismatch: ' + field)
    names = manifest['joint_names']
    if len(names) != 31 or len(set(names)) != 31:
        raise ValueError('Expected 31 unique runtime joint names')
    return manifest


def validate_drives(manifest, actual, tolerance=1e-4):
    if actual['joint_names'] != manifest['joint_names']:
        raise ValueError('Runtime joint order mismatch')
    differences = {}
    for key in ('default_q', 'default_dq', 'stiffness', 'damping', 'effort_limit'):
        a, b = manifest[key], actual[key]
        if len(a) != 31 or len(b) != 31:
            raise ValueError('Runtime vector length mismatch: ' + key)
        if not all(math.isfinite(v) for v in a + b):
            raise ValueError('Nonfinite runtime vector: ' + key)
        err = max(abs(x - y) for x, y in zip(a, b))
        differences[key] = err
        if err > tolerance:
            raise ValueError('Runtime drive mismatch: %s max_error=%g' % (key, err))
    return differences


def summarize(trace, warmup=2.0):
    if not trace:
        raise ValueError('No simulation samples')
    for row in trace:
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite or nonnumeric simulation sample')
    if any(b['t'] <= a['t'] for a, b in zip(trace, trace[1:])):
        raise ValueError('Nonmonotonic simulation time')
    selected = [r for r in trace if r['t'] >= warmup]
    start, end = trace[0], trace[-1]
    result = {
        'simulated_seconds': end['t'],
        'world_displacement_xy': [end['x'] - start['x'], end['y'] - start['y']],
        'final_yaw_deg': end['yaw_deg'],
        'yaw_range_deg': [min(r['yaw_deg'] for r in trace), max(r['yaw_deg'] for r in trace)],
        'minimum_height': min(r['height'] for r in trace),
        'maximum_tilt_deg': max(r['tilt_deg'] for r in trace),
        'warmup_seconds': warmup,
        'post_warmup_samples': len(selected),
    }
    if selected:
        result.update(
            mean_body_vx=sum(r['vx'] for r in selected) / len(selected),
            mean_signed_body_vy=sum(r['vy'] for r in selected) / len(selected),
            mean_absolute_body_vy=sum(abs(r['vy']) for r in selected) / len(selected),
            fraction_body_vy_negative=sum(r['vy'] < 0 for r in selected) / len(selected),
        )
    return result


def emit_report(report):
    raw = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > 32768:
        raise ValueError('Report exceeds bounded inline evidence size')
    print(json.dumps({'stage': 'straight_eval_report_evidence', 'report': report,
                      'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
                     allow_nan=False, separators=(',', ':')), flush=True)