"""CPU-only synthetic tests; no network, CUDA, Flux mutations, or checkpoints loaded."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from input_delivery import InputFailure, prepare_archive
spec=importlib.util.spec_from_file_location('hud04_worker_for_test',HERE/'train_resume.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class Response(io.BytesIO):
    def __init__(self,body,status=200,length=None):
        super().__init__(body);self.status=status
        self.headers={} if length is None else {'Content-Length':str(length)}


class Opener:
    def __init__(self,response=None,error=None):self.response=response;self.error=error;self.called=False
    def open(self,*args,**kwargs):
        self.called=True
        if self.error:raise self.error
        return self.response


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        scratch=HERE.parents[1]/'outputs';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='cpu-regression-',dir=scratch)
        self.addCleanup(self.temp.cleanup);self.work=Path(self.temp.name)
        self.body=b'private synthetic input, not a robot asset'
        self.config={'input_bytes':len(self.body),'input_sha256':hashlib.sha256(self.body).hexdigest()}
        self.events=[]
    def emit(self,stage,**kwargs):self.events.append({'stage':stage,**kwargs})
    def prepare(self,config=None,opener=None,clock=None):
        kw={} if clock is None else {'clock':clock}
        return prepare_archive(config or self.config,self.work,opener,self.emit,lambda *a:None,**kw)
    def local(self):
        p=self.work/'mounted.tar.gz';p.write_bytes(self.body);self.config['input_archive']=str(p);return p
    def failure(self,reason,call):
        with self.assertRaises(InputFailure) as e:call()
        self.assertEqual(e.exception.reason,reason)
    def test_local_success_does_not_use_network(self):
        p=self.local();op=Opener(error=AssertionError('Network must not run'))
        self.assertEqual(self.prepare(opener=op),p);self.assertFalse(op.called)
        self.assertEqual(self.events[-1]['stage'],'input_transfer_verified')
    def test_local_required_by_default(self):
        self.failure('local_input_required',self.prepare)
    def test_remote_requires_boolean_opt_in(self):
        self.config['allow_remote_input']='true';self.failure('local_input_required',self.prepare)
    def test_missing_local(self):
        self.config['input_archive']=str(self.work/'missing');self.failure('local_input_missing',self.prepare)
    def test_symlink_rejected(self):
        p=self.local();q=self.work/'link';q.symlink_to(p);self.config['input_archive']=str(q)
        self.failure('local_input_missing',self.prepare)
    def test_local_size_mismatch(self):
        self.local();self.config['input_bytes']+=1;self.failure('input_size_mismatch',self.prepare)
    def test_local_hash_mismatch(self):
        self.local();self.config['input_sha256']='0'*64;self.failure('input_sha256_mismatch',self.prepare)
    def remote(self):
        self.config.update(allow_remote_input=True,input_url='https://storage.googleapis.com/synthetic-test-only')
    def test_remote_success_and_progress(self):
        self.remote();p=self.prepare(opener=Opener(Response(self.body,length=len(self.body))))
        self.assertEqual(p.read_bytes(),self.body)
        self.assertIn('input_progress',[e['stage'] for e in self.events])
    def test_http_status_failure(self):
        self.remote();self.failure('input_http_status',lambda:self.prepare(opener=Opener(Response(b'',status=403))))
    def test_content_length_rejected_before_read(self):
        self.remote();self.failure('input_content_length_mismatch',lambda:self.prepare(opener=Opener(Response(self.body,length=999))))
    def test_oversize_rejected(self):
        self.remote();self.failure('input_size_exceeded',lambda:self.prepare(opener=Opener(Response(self.body+b'extra'))))
    def test_truncated_rejected(self):
        self.remote();self.failure('input_size_mismatch',lambda:self.prepare(opener=Opener(Response(self.body[:-1]))))
    def test_total_deadline_not_extended(self):
        self.remote();clock=iter([0,241]).__next__
        self.failure('input_deadline_exceeded',lambda:self.prepare(opener=Opener(Response(self.body)),clock=clock))
    def test_slow_download_stops_with_specific_reason(self):
        self.remote();self.config['input_bytes']=1_000_000;clock=iter([0,31]).__next__
        self.failure('input_rate_insufficient',lambda:self.prepare(opener=Opener(Response(b'x'*65536)),clock=clock))
    def test_transport_error_never_exposes_url(self):
        self.remote()
        with self.assertRaises(InputFailure) as e:self.prepare(opener=Opener(error=OSError('https://private.invalid/?signature=hidden')))
        self.assertEqual(e.exception.reason,'input_transport_failure')
        self.assertNotIn('signature',str(e.exception)+json.dumps(e.exception.metrics))
    def run_main(self,path,parent=None):
        out=io.StringIO()
        with patch.object(sys,'argv',['train_resume.py','--private-config-file',str(path)]),contextlib.redirect_stdout(out):
            if parent:
                with patch.object(worker,'parent',side_effect=parent):code=worker.main()
            else:code=worker.main()
        return code,json.loads(out.getvalue())
    def test_private_config_permissions(self):
        p=self.work/'config.json';p.write_text('{}');p.chmod(0o644)
        code,event=self.run_main(p);self.assertEqual(code,1)
        self.assertEqual(event['reason'],'private_config_permissions_too_broad')
    def test_private_config_must_be_object(self):
        p=self.work/'config.json';p.write_text('[]');p.chmod(0o600)
        code,event=self.run_main(p);self.assertEqual(event['reason'],'private_config_not_an_object')
    def test_generic_exception_has_location_not_secret_message(self):
        p=self.work/'config.json';p.write_text('{"run_id":"test0001"}');p.chmod(0o600)
        code,event=self.run_main(p,ValueError('sensitive-url-must-not-be-logged'))
        self.assertEqual(code,1);self.assertEqual(event['reason'],'unclassified_bootstrap_exception')
        self.assertIn('line',event['location']);self.assertNotIn('sensitive',json.dumps(event))
    def test_run_id_path_traversal_rejected(self):
        p=self.work/'config.json';p.write_text('{"run_id":"../other"}');p.chmod(0o600)
        code,event=self.run_main(p);self.assertEqual(event['reason'],'invalid_run_id')
    def test_private_config_cannot_choose_report_path(self):
        p=self.work/'config.json';p.write_text(json.dumps({'run_id':'test0001','_work_dir':'/untrusted/path'}));p.chmod(0o600)
        def check(config):
            self.assertNotIn('_work_dir',config)
            return 0
        code,event=None,None
        with patch.object(sys,'argv',['train_resume.py','--private-config-file',str(p)]),patch.object(worker,'parent',side_effect=check):
            self.assertEqual(worker.main(),0)
    def test_headless_metadata_excludes_keyboard_and_mujoco(self):
        with patch('setuptools.setup') as setup,patch('setuptools.find_packages',return_value=[]):
            runpy.run_path(str(HERE.parents[1]/'setup.py'))
        kw=setup.call_args.kwargs
        for package in ['pynput','mujoco==3.3.2','mujoco-python-viewer']:
            self.assertNotIn(package,kw['install_requires']);self.assertIn(package,kw['extras_require']['sim2sim'])
    def test_url_allowlist_remains_enforced(self):
        with self.assertRaises(ValueError):worker.validate_url('https://unapproved.invalid/test',1)


if __name__=='__main__':unittest.main(verbosity=2)