"""Single bounded HU_D04 AMP continuation; no account credentials required."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import urlsplit, parse_qs
from urllib.request import Request, build_opener, HTTPRedirectHandler

SOURCE_COMMIT = '3e444244d41febad2e2827168c178f6f68d04ea4'
CHECKPOINT_SHA = 'f3d91b4b0e536c9151d5d27c6a6adc30e44e968010547f3ffec359f16af856c7'

def emit(stage, **kw):
    print(json.dumps(dict(stage='hud04_resume_'+stage, **kw), allow_nan=False), flush=True)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('Redirect forbidden')

def validate_url(url, minimum_remaining):
    from datetime import datetime, timezone
    p=urlsplit(url)
    if p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.username or p.password or p.port not in (None,443):
        raise ValueError('Unapproved storage host')
    q=parse_qs(p.query)
    if 'X-Goog-Date' in q:
        expires=datetime.strptime(q['X-Goog-Date'][0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()+int(q['X-Goog-Expires'][0])
    else:
        expires=int(q['Expires'][0])
    if expires-time.time()<minimum_remaining: raise ValueError('Expired storage capability')

def runtime_environment(root):
    exact={'PATH','HOME','USER','LOGNAME','LANG','LC_ALL','LD_LIBRARY_PATH','LIBRARY_PATH','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX','DISPLAY','WAYLAND_DISPLAY','XDG_RUNTIME_DIR','ACCEPT_EULA','PRIVACY_CONSENT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','EXP_PATH','ISAACSIM_PATH'}
    prefixes=('CUDA_','NVIDIA_','ISAAC_','OMNI_','CARB_','KIT_','PHYSX_')
    env={k:v for k,v in os.environ.items() if (k in exact or k.startswith(prefixes)) and not any(x in k.upper() for x in ('KEY','TOKEN','PASSWORD','SECRET'))}
    env['PYTHONPATH']=str(root/'rsl_rl')+os.pathsep+str(root)+os.pathsep+env.get('PYTHONPATH','')
    env.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    return env

def extract_inputs(archive, root):
    seen=set(); count=0; expanded=0
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            p=PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in seen:
                raise ValueError('Invalid archive member')
            seen.add(member.name)
            if member.name.startswith('assets/HU_D04_description/'):
                target=root/'humanoid-description'/Path(*p.parts[1:])
                expected=None  # Full archive identity was verified before extraction.
            elif member.name=='checkpoint/model_15500.pt':
                target=root/'resume-inputs/model_15500.pt'; expected=CHECKPOINT_SHA
            else:raise ValueError('Unexpected input member')
            expanded+=member.size
            if expanded>180_000_000:raise ValueError('Input expansion exceeds bound')
            target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src, target.open('xb') as out:
                while True:
                    b=src.read(1024*1024)
                    if not b:break
                    out.write(b)
            if expected and sha(target)!=expected:raise ValueError('Input hash mismatch')
            count+=1
    if count!=119:raise ValueError('Missing input members')
    emit('inputs_verified',files=count,checkpoint_sha256=CHECKPOINT_SHA)

def train_child(work):
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path[:0]=[str(root/'rsl_rl'),str(root)]
    from isaaclab.app import AppLauncher
    emit('simulator_initializing')
    app=AppLauncher(headless=True,fast_shutdown=False).app
    env=None; runner=None
    report={'source_checkpoint_sha256':CHECKPOINT_SHA,'source_commit':SOURCE_COMMIT,'updates':0,'success':False}
    started=time.monotonic()
    class SoftStop(Exception):pass
    try:
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.envs import task_registry
        from rsl_rl.runners import AmpOnPolicyRunner
        emit('runtime',python=sys.version.split()[0],torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0))
        cfg,agent=task_registry.get_cfgs('hu_d04_run')
        cfg.scene.num_envs=4096
        cfg.scene.seed=agent.seed
        cfg.commands.ranges.lin_vel_x=(1.2,2.0)
        cfg.commands.debug_vis=False
        agent.logger='tensorboard'
        agent.max_iterations=2000
        agent.save_interval=100
        logdir=work/'training'; logdir.mkdir(exist_ok=True)
        dump_yaml(str(logdir/'params/env.yaml'),cfg)
        dump_yaml(str(logdir/'params/agent.yaml'),agent)
        class BoundedRunner(AmpOnPolicyRunner):
            def log(self,locs,*args,**kwargs):
                super().log(locs,*args,**kwargs)
                report['updates']+=1
                n=report['updates']
                speed=float(self.env.robot.data.root_lin_vel_b[:,0].mean().item())
                command=float(self.env.command_generator.command[:,0].mean().item())
                loss={k:float(v) for k,v in locs['loss_dict'].items()}
                metrics={'updates':n,'iteration':int(self.current_learning_iteration),'train_mean_forward_velocity':speed,'train_mean_command_x':command,'learning_rate':float(self.alg.learning_rate),'losses':loss}
                with (work/'progress.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
                report['last_metrics']=metrics
                if n<=3 or n%10==0:emit('update',**metrics)
                # Soft deadline is checked only after a complete optimizer update.
                if time.monotonic()>=soft_deadline:
                    report['stop_reason']='soft_deadline'; raise SoftStop()
        env=task_registry.get_task_class('hu_d04_run')(cfg,True)
        emit('environment_ready',num_envs=env.num_envs,actions=env.num_actions)
        runner=BoundedRunner(env,agent.to_dict(),log_dir=str(logdir),device=agent.device)
        cp=root/'resume-inputs/model_15500.pt'
        if sha(cp)!=CHECKPOINT_SHA:raise ValueError('Checkpoint identity changed')
        runner.load(str(cp),load_optimizer=True)
        initial=int(runner.current_learning_iteration)
        if initial!=15500 or not runner.alg.optimizer.state:raise ValueError('Full optimizer resume failed')
        # Preserve loaded optimizer LR instead of resetting adaptive PPO to config LR.
        runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
        emit('checkpoint_loaded',iteration=initial,optimizer_states=len(runner.alg.optimizer.state),learning_rate=runner.alg.learning_rate)
        soft_deadline=float((work/'deadline.txt').read_text())
        try:
            runner.learn(num_learning_iterations=2000,init_at_random_ep_len=True)
            report['stop_reason']='update_limit'
        except SoftStop:
            pass
        runner.current_learning_iteration=initial+report['updates']
        final=work/'final_checkpoint.pt'
        runner.save(str(final))
        report.update(success=report['updates']>0,final_iteration=runner.current_learning_iteration,checkpoint_bytes=final.stat().st_size,checkpoint_sha256=sha(final),training_wall_seconds=time.monotonic()-started)
        # Re-open the just-saved trusted checkpoint to check the full resume state.
        saved=torch.load(final,map_location='cpu',weights_only=False)
        required={'model_state_dict','optimizer_state_dict','discriminator_state_dict','amp_normalizer','iter'}
        if not required.issubset(saved) or saved['iter']!=runner.current_learning_iteration:raise ValueError('Final checkpoint incomplete')
        report['checkpoint_structure_verified']=True
        emit('checkpoint_saved',**{k:report[k] for k in ('updates','final_iteration','checkpoint_bytes','checkpoint_sha256')})
    except BaseException as error:
        report.update(success=False,error_type=type(error).__name__)
        emit('child_failed',error_type=type(error).__name__)
        import traceback
        traceback.print_exc()
        raise
    finally:
        (work/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None and hasattr(env,'close'):env.close()
        app.close(wait_for_replicator=False)
        emit('cleanup_completed')
    return 0

def parent(config):
    root=Path(__file__).resolve().parents[2]
    if time.time()>config['start_before']:raise ValueError('Launch window expired')
    validate_url(config['input_url'],240)
    validate_url(config['output_url'],4200)
    os.chdir(root)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    emit('started',commit=commit,source_commit=SOURCE_COMMIT,max_updates=2000,wall_seconds=3600)
    work=root/'outputs'/('speed-resume-'+config['run_id']); work.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    # Independent hard deadline prevents an unresponsive simulator from running indefinitely.
    children=[]
    def hard_stop():
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        os._exit(124)
    timer=threading.Timer(3590,hard_stop);timer.daemon=True;timer.start()
    opener=build_opener(NoRedirect())
    archive=work/'inputs.tar.gz'; size=0
    with opener.open(Request(config['input_url'],headers={'Accept-Encoding':'identity'}),timeout=30) as response,archive.open('xb') as out:
        if response.status!=200:raise ValueError('Input download failed')
        while True:
            block=response.read(1024*1024)
            if not block:break
            size+=len(block)
            if size>config['input_bytes'] or time.monotonic()-started>240:raise ValueError('Input download bound exceeded')
            out.write(block)
    if size!=config['input_bytes'] or sha(archive)!=config['input_sha256']:raise ValueError('Archive integrity failed')
    extract_inputs(archive,root)
    (work/'deadline.txt').write_text(str(started+3300))
    env=runtime_environment(root)
    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child',str(work)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(proc)
    seen=set()
    def read():
        with (work/'training.log').open('wb') as log:
            for line in iter(proc.stdout.readline,b''):
                log.write(line)
                if line.startswith(b'{') and len(line)<65536:
                    try:
                        event=json.loads(line)
                        if event.get('stage','').startswith('hud04_resume_'):
                            seen.add(event['stage']);print(json.dumps(event,allow_nan=False),flush=True)
                    except (ValueError,AttributeError):pass
    reader=threading.Thread(target=read,daemon=True);reader.start()
    timed_out=False
    try:proc.wait(timeout=max(1,started+3400-time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    reader.join(timeout=5)
    result=work/'training-results.tar.gz'
    report_path=work/'report.json'
    if report_path.exists():report=json.loads(report_path.read_text())
    else:report={'success':False,'updates':0,'error_type':'missing_child_report'}
    report.update(child_returncode=proc.returncode,timed_out=timed_out,worker_commit=commit,cleanup_verified='hud04_resume_cleanup_completed' in seen)
    report['success']=bool(report.get('success') and proc.returncode==0 and not timed_out and report['cleanup_verified'])
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    with tarfile.open(result,'w:gz') as t:
        for name in ('final_checkpoint.pt','report.json','progress.jsonl','training.log'):
            if (work/name).exists():t.add(work/name,arcname=name)
        params=work/'training/params'
        if params.exists():t.add(params,arcname='params')
    if result.stat().st_size>100_000_000:raise ValueError('Result size exceeds bound')
    headers={}
    for entry in config['output_headers']:
        k,sep,v=entry.partition(':')
        if not sep:raise ValueError('Invalid upload header')
        headers[k.strip()]=v.strip()
    headers['Content-Length']=str(result.stat().st_size)
    with result.open('rb') as content:
        with opener.open(Request(config['output_url'],data=content,headers=headers,method='PUT'),timeout=90) as response:
            if response.status not in (200,201,204):raise ValueError('Result upload failed')
    emit('finished',success=report['success'],updates=report.get('updates',0),result_bytes=result.stat().st_size,result_sha256=sha(result),uploaded=True,wall_seconds=time.monotonic()-started)
    timer.cancel()
    return 0 if report['success'] else 1

def main():
    if len(sys.argv)==3 and sys.argv[1]=='--child':return train_child(Path(sys.argv[2]))
    parser=argparse.ArgumentParser()
    parser.add_argument('--private-config-b64',required=True)
    args=parser.parse_args()
    config=json.loads(base64.b64decode(args.private_config_b64,validate=True))
    sys.argv[:]=sys.argv[:1]
    try:return parent(config)
    except Exception as error:
        emit('bootstrap_failed',error_type=type(error).__name__)
        return 1

if __name__=='__main__':raise SystemExit(main())