"""One evaluation only; fixed inputs; no optimizer, credentials or network fetches."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FINAL_SHA='44328830c3cedd004597c4b893c0479fd744d65d9bef465e10cb5a393ae64e21'

def emit(stage,**kw):
    print(json.dumps({'stage':'hud04_native_eval_'+stage,**kw},allow_nan=False),flush=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def locate_final_checkpoint():
    # Only user-application input locations configured in this task, no host probing.
    directory=ROOT/'native-eval-input'
    candidates=[]
    if directory.is_file():candidates.append(directory)
    elif directory.is_dir():
        for p in directory.rglob('*'):
            if p.is_file():candidates.append(p)
            if len(candidates)>32:raise ValueError('too_many_native_input_files')
    for p in candidates:
        if p.is_symlink():continue
        if p.stat().st_size==26811026 and sha(p)==FINAL_SHA:return p.resolve()
    raise ValueError('platform_checkpoint_input_missing_or_hash_mismatch')

def main():
    began=time.monotonic();children=[]
    def kill_child(sig):
        for p in children:
            if p.poll() is None:
                try:os.killpg(p.pid,sig)
                except ProcessLookupError:pass
    def hard_deadline():
        emit('hard_deadline',worker_wall_seconds=720)
        kill_child(signal.SIGKILL)
        os._exit(124)
    timer=threading.Timer(720,hard_deadline);timer.daemon=True;timer.start()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    import coldplay.model_monitor as monitor
    assert sha(Path(monitor.__file__))=='ce2cfa301ade8fcb9bc7373e5b5c50843b1f23d311056e46b3427f601b818740','official_model_monitor_revision_changed'
    emit('started',commit=commit,training_updates=0,worker_wall_seconds=720,parallel_envs=18)
    work=ROOT/'outputs/native-eval-17309';work.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'tools/hud04_speed_resume'))
    from train_resume import extract_inputs,runtime_environment
    from input_delivery import assemble_git_archive
    checkpoint=locate_final_checkpoint()
    emit('private_checkpoint_verified',checkpoint_sha256=FINAL_SHA,source_task='TASK_20261119_441')
    archive=assemble_git_archive(ROOT/'data/hud04_resume_15500/manifest.json',work,69716909,'0078b6fdfd7f8a9fb083614e371f027134b20fde8ee9bcaa7271a052d0a8ce00',emit)
    extract_inputs(archive,ROOT)
    usd=ROOT/'humanoid-description/HU_D04_description/usd/HU_D04_01.usd'
    assert sha(usd)=='9cd8cf50aa56e09a8aea8bb9a3fdec15af2a4dad622882c1d6f39fd3e967baa0'
    env=runtime_environment(ROOT);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,str(HERE/'evaluate_native.py'),'--repo',str(ROOT),'--bundle',str(HERE),'--usd',str(usd),'--output',str(work/'evaluation'),'--final-checkpoint',str(checkpoint),'--allow-gpu-eval']
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(proc)
    stages=[]
    def read():
        with (work/'worker.log').open('xb') as log:
            for line in iter(proc.stdout.readline,b''):
                log.write(line)
                # No secrets are passed to the child. Keep diagnostic tracebacks,
                # but do not echo arbitrary simulator URLs or whole configs.
                text=line.decode(errors='replace').strip()
                if text.startswith('{') and len(line)<32768:
                    try:
                        event=json.loads(text)
                        if 'stage' in event:stages.append(event['stage'])
                        if 'stage' in event or 'step' in event or 'completed' in event:print(json.dumps(event),flush=True)
                    except (ValueError,TypeError):pass
                elif text.startswith(('Traceback (','AssertionError','AttributeError','ModuleNotFoundError','TypeError','ValueError','RuntimeError','  File ')):
                    if 'http' not in text:print(text[:1000],flush=True)
    reader=threading.Thread(target=read,daemon=True);reader.start()
    timed_out=False;result_seen=None
    result=work/'evaluation/native_results.json'
    try:
        while proc.poll() is None:
            elapsed=time.monotonic()-began
            if result.is_file() and result_seen is None:result_seen=time.monotonic()
            if elapsed>=680 or (result_seen is not None and time.monotonic()-result_seen>30):raise subprocess.TimeoutExpired(cmd,elapsed)
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
    except subprocess.TimeoutExpired:
        timed_out=True;kill_child(signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:kill_child(signal.SIGKILL);proc.wait(timeout=5)
    reader.join(timeout=3)
    report={'commit':commit,'training_updates':0,'child_returncode':proc.returncode,'timed_out':timed_out,'native_result_present':result.is_file(),'cleanup_verified':'hud04_native_eval_cleanup_completed' in stages,'worker_wall_seconds':time.monotonic()-began}
    if result.is_file():report['evaluation']=json.loads(result.read_text())
    frames=work/'evaluation/frames.jsonl'
    if frames.is_file():report['trace_frames']=[json.loads(line) for line in frames.read_text().splitlines()]
    report['execution_success']=bool(result.is_file() and proc.returncode==0 and not timed_out and report['cleanup_verified'])
    # The official platform SDK monitors .pt. This is a report, NOT a checkpoint.
    import torch
    out=work/'native_eval_report.pt';torch.save(report,out)
    emit('report_saved',report_bytes=out.stat().st_size,report_sha256=sha(out),**{k:v for k,v in report.items() if k!='trace_frames'})
    timer.cancel()
    return 0 if report['execution_success'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        emit('bootstrap_failed',error_type=type(e).__name__,reason=str(e) if isinstance(e,ValueError) and str(e).replace('_','').isalnum() else 'see_private_worker_log')
        raise