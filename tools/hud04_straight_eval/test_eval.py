import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import eval_core as core
import supervisor


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'checkpoint'
        self.policy = self.root / 'policy'
        self.checkpoint.write_bytes(b'synthetic checkpoint, not weights')
        self.policy.write_bytes(b'synthetic policy, not weights')
        self.manifest = dict(checkpoint_sha256=core.sha256(self.checkpoint),
                             policy_sha256=core.sha256(self.policy),
                             joint_names=['joint_%02d' % i for i in range(31)])
        self.manifest.update({k: [1.0]*31 for k in
                             ('default_q', 'default_dq', 'stiffness', 'damping', 'effort_limit')})
        self.path = self.root / 'manifest.json'
        self.path.write_text(json.dumps(self.manifest))

    def test_valid_identity(self):
        self.assertEqual(core.verify_inputs(self.path, self.checkpoint, self.policy), self.manifest)

    def test_changed_checkpoint_rejected(self):
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            core.verify_inputs(self.path, self.checkpoint, self.policy)

    def test_changed_policy_rejected(self):
        self.policy.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            core.verify_inputs(self.path, self.checkpoint, self.policy)

    def test_duplicate_joint_rejected(self):
        self.manifest['joint_names'][0] = self.manifest['joint_names'][1]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            core.verify_inputs(self.path, self.checkpoint, self.policy)

    def test_equal_drives(self):
        self.assertTrue(all(v == 0 for v in core.validate_drives(self.manifest, self.manifest).values()))

    def test_each_drive_mismatch_rejected(self):
        for key in ('default_q', 'default_dq', 'stiffness', 'damping', 'effort_limit'):
            with self.subTest(key=key):
                actual = copy.deepcopy(self.manifest)
                actual[key][0] += 0.1
                with self.assertRaises(ValueError):
                    core.validate_drives(self.manifest, actual)

    def test_joint_order_rejected(self):
        actual = copy.deepcopy(self.manifest)
        actual['joint_names'].reverse()
        with self.assertRaises(ValueError):
            core.validate_drives(self.manifest, actual)

    def test_nonfinite_drive_rejected(self):
        actual = copy.deepcopy(self.manifest)
        actual['damping'][0] = float('nan')
        with self.assertRaises(ValueError):
            core.validate_drives(self.manifest, actual)

    def trace(self):
        return [dict(t=t, x=t, y=-t/2, yaw_deg=-t, height=1.0, tilt_deg=t,
                     vx=1.0, vy=vy) for t, vy in [(0, 0), (1, 99), (2, -0.4), (3, 0.2)]]

    def test_signed_metrics_and_warmup(self):
        result = core.summarize(self.trace())
        self.assertAlmostEqual(result['mean_signed_body_vy'], -0.1)
        self.assertAlmostEqual(result['mean_absolute_body_vy'], 0.3)
        self.assertEqual(result['fraction_body_vy_negative'], 0.5)
        self.assertEqual(result['world_displacement_xy'], [3, -1.5])
        self.assertEqual(result['post_warmup_samples'], 2)

    def test_early_termination_no_fake_mean(self):
        result = core.summarize(self.trace()[:2])
        self.assertEqual(result['post_warmup_samples'], 0)
        self.assertNotIn('mean_body_vx', result)

    def test_empty_nonfinite_nonmonotonic_rejected(self):
        bad = self.trace()
        bad[-1]['vy'] = float('nan')
        for trace in ([], bad, list(reversed(self.trace()))):
            with self.subTest(trace=str(trace)[:50]), self.assertRaises(ValueError):
                core.summarize(trace)

    def test_canonical_evidence(self):
        capture = io.StringIO()
        report = {'z': 1, 'a': {'b': 2}}
        with contextlib.redirect_stdout(capture):
            core.emit_report(report)
        evidence = json.loads(capture.getvalue())
        raw = json.dumps(report, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(evidence['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(evidence['bytes'], len(raw))

    def test_oversize_and_nonfinite_evidence_rejected(self):
        for report in ({'payload': 'x'*32769}, {'nan': float('nan')}):
            with self.assertRaises(ValueError):
                core.emit_report(report)


class FakeChild:
    pid = 123456

    def __init__(self, events, running=False):
        self.stdout = io.StringIO('\n'.join(json.dumps(e) for e in events) + '\n')
        self.returncode = None if running else 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = -15 if self.returncode is None else self.returncode
        return self.returncode


class SupervisorTests(unittest.TestCase):
    def run_case(self, events, running=False, clocks=None):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.receipt = Path(self.temp.name) / 'receipt.json'
        child = FakeChild(events, running)
        arguments = ['supervisor.py', '--python', sys.executable, '--receipt', str(self.receipt), '--', '--fake']
        with patch.object(sys, 'argv', arguments), patch.object(supervisor.subprocess, 'Popen', return_value=child) as popen, \
             patch.object(supervisor.os, 'killpg') as kill, patch.dict(supervisor.os.environ, {'ARBITRARY_PERSONAL_CREDENTIAL': 'synthetic'}), \
             patch.object(supervisor.time, 'monotonic', side_effect=clocks) if clocks else contextlib.nullcontext(), \
             contextlib.redirect_stdout(io.StringIO()):
            result = supervisor.main()
            self.assertNotIn('ARBITRARY_PERSONAL_CREDENTIAL', popen.call_args.kwargs['env'])
            calls = kill.call_count
        return result, json.loads(self.receipt.read_text()), calls

    def valid_events(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            core.emit_report({'synthetic': True})
        return [{'stage': 'straight_eval_rollout_started'}, json.loads(capture.getvalue()),
                {'stage': 'straight_eval_cleanup_started'},
                {'stage': 'straight_eval_cleanup_completed', 'cleanup_ok': True}]

    def test_valid_terminal_evidence_required(self):
        result, receipt, _ = self.run_case(self.valid_events())
        self.assertEqual(result, 0)
        self.assertTrue(receipt['report_verified'])
        self.assertTrue(receipt['cleanup_completed'])

    def test_missing_cleanup_is_failure(self):
        result, receipt, _ = self.run_case(self.valid_events()[:-1])
        self.assertEqual(result, 1)
        self.assertFalse(receipt['cleanup_completed'])

    def test_zero_exit_without_report_is_failure(self):
        result, receipt, _ = self.run_case([])
        self.assertEqual(result, 1)
        self.assertFalse(receipt['report_verified'])

    def test_initialization_watchdog_stops_own_child(self):
        result, receipt, calls = self.run_case([], running=True, clocks=[0, 301, 302])
        self.assertEqual(result, 1)
        self.assertTrue(receipt['watchdog_triggered'])
        self.assertEqual(calls, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)