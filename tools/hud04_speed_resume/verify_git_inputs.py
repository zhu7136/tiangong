"""CPU-only Git input verification. No network, pickle deserialization or GPU."""
import argparse
import json
from pathlib import Path
from input_delivery import prepare_archive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='New local directory for verified joined archive')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    path=root/'data/hud04_resume_15500/manifest.json'
    # The original archive identity is fixed outside the mutable manifest too.
    config={'git_input_manifest':str(path),'input_bytes':69716909,
            'input_sha256':'0078b6fdfd7f8a9fb083614e371f027134b20fde8ee9bcaa7271a052d0a8ce00'}
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw)),flush=True)
    def deny(*args,**kwargs):raise AssertionError('Network not allowed in Git preflight')
    archive=prepare_archive(config,out,None,emit,deny)
    emit('git_preflight_passed',archive_bytes=archive.stat().st_size,checkpoint_deserialized=False,gpu_used=False)

if __name__=='__main__':main()