"""One approved Flux simulation trial. No credentials or model weights in Git."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CHILDREN=[]

def emit(stage,**kw):print(json.dumps({'stage':'hud04_limit8_'+stage,**kw},allow_nan=False),flush=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def kill(proc,sig):
    try:os.killpg(proc.pid,sig)
    except ProcessLookupError:pass

def run_child(name,cmd,env,work,seconds):
    emit('child_start',child=name,child_wall_limit_seconds=seconds)
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append(proc)
    def reader():
        with (work/(name+'.log')).open('xb') as log:
            for line in iter(proc.stdout.readline,b''):
                log.write(line)
                text=line.decode(errors='replace').strip()
                if text.startswith('{') and len(line)<32768:
                    try:
                        event=json.loads(text)
                        if any(k in event for k in ['stage','step','completed']):print(json.dumps(event),flush=True)
                    except (ValueError,TypeError):pass
                elif text.startswith(('Traceback (','AssertionError','AttributeError','ModuleNotFoundError','TypeError','ValueError','RuntimeError','ImportError','  File ')):
                    if 'http' not in text:print(text[:1000],flush=True)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    try:code=proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        kill(proc,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:kill(proc,signal.SIGKILL);proc.wait(timeout=5)
        raise TimeoutError(name+'_hard_wall_limit')
    thread.join(timeout=3)
    if code:raise RuntimeError(name+'_child_failed_'+str(code))
    emit('child_completed',child=name,returncode=code)

def main():
    began=time.monotonic()
    def deadline():
        emit('worker_hard_deadline',worker_wall_seconds=3500)
        for proc in CHILDREN:
            if proc.poll() is None:kill(proc,signal.SIGKILL)
        os._exit(124)
    timer=threading.Timer(3500,deadline);timer.daemon=True;timer.start()
    import coldplay.model_monitor as monitor
    assert sha(Path(monitor.__file__))=='ce2cfa301ade8fcb9bc7373e5b5c50843b1f23d311056e46b3427f601b818740'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    emit('started',commit=commit,gpu_count=1,hip_knee_limit=8.,max_optimizer_wall_seconds=1800,worker_wall_seconds=3500,simulation_only=True)
    work=ROOT/'outputs/limit8-trial';work.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'tools/hud04_speed_resume'))
    from train_resume import extract_inputs,runtime_environment,CHECKPOINT_SHA
    from input_delivery import assemble_git_archive
    archive=assemble_git_archive(ROOT/'data/hud04_resume_15500/manifest.json',work,69716909,'0078b6fdfd7f8a9fb083614e371f027134b20fde8ee9bcaa7271a052d0a8ce00',emit)
    extract_inputs(archive,ROOT)
    checkpoint=ROOT/'resume-inputs/model_15500.pt'
    usd=ROOT/'humanoid-description/HU_D04_description/usd/HU_D04_01.usd'
    assert sha(checkpoint)==CHECKPOINT_SHA
    assert sha(usd)=='9cd8cf50aa56e09a8aea8bb9a3fdec15af2a4dad622882c1d6f39fd3e967baa0'
    bundle=work/'bundle';bundle.mkdir()
    source_bundle=ROOT/'tools/hud04_native_eval'
    contract=json.loads((source_bundle/'deployment_contract.json').read_text())
    contract['checkpoint_sha256']=CHECKPOINT_SHA;contract.pop('onnx_sha256',None)
    (bundle/'deployment_contract.json').write_text(json.dumps(contract,indent=2))
    (bundle/'native_eval_plan.json').write_bytes((source_bundle/'native_eval_plan.json').read_bytes())
    env=runtime_environment(ROOT);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    base=[sys.executable,str(source_bundle/'evaluate_native.py'),'--repo',str(ROOT),'--bundle',str(bundle),'--usd',str(usd),'--allow-gpu-eval','--hip-knee-limit','8','--process-exit-after-result']
    report={'commit':commit,'simulation_only':True,'hip_knee_limit':8.,'source_iteration':15500,'source_checkpoint_sha256':CHECKPOINT_SHA,'success':False}
    try:
        run_child('preflight',base+['--output',str(work/'preflight'),'--final-checkpoint',str(checkpoint),'--final-iteration','15500'],env,work,420)
        preflight=json.loads((work/'preflight/native_results.json').read_text())
        assert len(preflight['results'])==18 and preflight['hip_knee_limit']==8.
        report['preflight']=preflight
        emit('preflight_verified',hip_knee_limit=8.,survived=sum(r['survived_12s'] for r in preflight['results']),rollouts=18,not_a_training_run=True)
        run_child('training',[sys.executable,str(HERE/'train_limit8.py'),'--repo',str(ROOT),'--output',str(work/'train'),'--seconds','1800'],env,work,2100)
        training=json.loads((work/'train/training_result.json').read_text())
        report['training']=training
        final=work/'train/final_checkpoint.pt'
        assert sha(final)==training['final_checkpoint_sha256'] and training['updates']>0
        contract['checkpoint_sha256']=training['final_checkpoint_sha256']
        (bundle/'deployment_contract.json').write_text(json.dumps(contract,indent=2))
        run_child('evaluation',base+['--output',str(work/'evaluation'),'--final-checkpoint',str(final),'--final-iteration',str(training['final_iteration'])],env,work,420)
        report['evaluation']=json.loads((work/'evaluation/native_results.json').read_text())
        report['trace_frames']=[json.loads(line) for line in (work/'evaluation/frames.jsonl').read_text().splitlines()]
        report['training_metrics']=[json.loads(line) for line in (work/'train/training_metrics.jsonl').read_text().splitlines()]
        report['success']=True
    except Exception as e:
        report['failure']={'type':type(e).__name__,'reason':str(e) if isinstance(e,(TimeoutError,RuntimeError)) and str(e).replace('_','').isalnum() else 'see_private_child_log'}
        emit('failed',**report['failure'])
        raise
    finally:
        report['worker_wall_seconds']=time.monotonic()-began
        import torch
        path=work/'limit8_trial_report.pt';torch.save(report,path)
        emit('report_saved',report_bytes=path.stat().st_size,report_sha256=sha(path),success=report['success'],worker_wall_seconds=report['worker_wall_seconds'])
        timer.cancel()
    emit('completed',success=True,final_iteration=training['final_iteration'],updates=training['updates'],checkpoint_sha256=training['final_checkpoint_sha256'])
    return 0

if __name__=='__main__':sys.exit(main())