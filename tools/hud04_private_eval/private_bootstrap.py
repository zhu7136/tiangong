"""Private-task config bootstrap: bounded download, verified inputs, inference only.

Never accepts account API keys or prints input URLs/configuration. No provisioning.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import urlsplit, parse_qs
from urllib.request import Request, build_opener, HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def emit(stage, **fields):
    print(json.dumps(dict(stage='hud04_private_' + stage, **fields), allow_nan=False), flush=True)


def validate_config(config, now=None):
    if config.get('schema_version') != 1 or config.get('enabled') is not True:
        raise ValueError('Private configuration is not enabled')
    if config['mode'] not in ('preflight', 'paired'):
        raise ValueError('Invalid mode')
    if config['archive_bytes'] != 101291988 or config['archive_sha256'] != '5ad9383f74968fae4740536674ccb049c37746fb656acb116413767cd60135d0':
        raise ValueError('Unexpected input archive')
    now = time.time() if now is None else now
    if now > config['start_before_utc_epoch'] or config['start_before_utc_epoch'] - now > 1800:
        raise ValueError('Launch window exceeded')
    parsed = urlsplit(config['archive_url'])
    if parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com' or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Unapproved input host')
    query = parse_qs(parsed.query)
    signed = datetime.datetime.strptime(query['X-Goog-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    expires = signed + int(query['X-Goog-Expires'][0])
    # Download, 1800-second evaluation and cleanup reserve must all fit AFTER queueing.
    if expires < config['start_before_utc_epoch'] + 2400:
        raise ValueError('Insufficient signed-input lifetime')
    if set(config) != {'schema_version','enabled','mode','archive_bytes','archive_sha256','archive_url','start_before_utc_epoch'}:
        raise ValueError('Unexpected private configuration fields')
    return expires


def download(config, destination):
    started = time.monotonic()
    opener = build_opener(NoRedirect())
    request = Request(config['archive_url'], headers={'Accept-Encoding': 'identity'})
    digest, size = hashlib.sha256(), 0
    with opener.open(request, timeout=20) as response, destination.open('xb') as output:
        if response.status != 200:
            raise ValueError('Unexpected download response')
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > config['archive_bytes'] or time.monotonic()-started > 180:
                raise ValueError('Download bound exceeded')
            digest.update(block)
            output.write(block)
    if size != config['archive_bytes'] or digest.hexdigest() != config['archive_sha256']:
        raise ValueError('Input archive identity mismatch')
    emit('download_verified', bytes=size, seconds=round(time.monotonic()-started,2))


def unpack(archive_path, target):
    target.mkdir(exist_ok=False)
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != 134 or sum(x.size for x in members) > 400_000_000:
            raise ValueError('Unexpected archive inventory')
        names = set()
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in names:
                raise ValueError('Unsafe archive member')
            names.add(member.name)
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with output.open('xb') as file:
                while True:
                    block = source.read(1024*1024)
                    if not block:
                        break
                    file.write(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default=str(Path(__file__).with_name('private_config.json')))
    args = parser.parse_args()
    config = json.loads(Path(args.config_file).read_text())
    validate_config(config)
    emit('config_verified', mode=config['mode'], optimizer_updates=0)
    work = Path.cwd() / ('hud04-private-' + str(time.time_ns()))
    work.mkdir(exist_ok=False)
    children = []
    def hard_deadline():
        for process in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os._exit(124)
    timer = threading.Timer(2160, hard_deadline)
    timer.daemon = True
    timer.start()
    source = Path(__file__).resolve().parents[2]
    training = work / 'training-source'
    expected = '3e444244d41febad2e2827168c178f6f68d04ea4'
    subprocess.run(['git','-C',str(source),'worktree','add','--detach',str(training),expected],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    archive_path = work / 'inputs.tar.gz'
    download(config, archive_path)
    inputs = work / 'inputs'
    unpack(archive_path, inputs)
    sys.path.insert(0, str(inputs / 'evaluator'))
    from materialize_inputs import materialize
    bundle = materialize(inputs, training)
    bundle_path = work / 'inputs.json'
    bundle_path.write_text(json.dumps(bundle))
    emit('inputs_verified', files=134, checkpoints=['14100','13000'])
    if config['mode'] == 'preflight':
        emit('complete', runtime_validated=False, optimizer_updates=0)
        return 0
    from gpu_probe import runtime_environment
    command = [sys.executable, str(inputs/'evaluator/paired_eval.py'), '--bundle',str(bundle_path),
               '--isaac-python',sys.executable,'--output',str(work/'results'),'--wall-seconds','1800']
    child = subprocess.Popen(command, env=runtime_environment(os.environ),start_new_session=True)
    children.append(child)
    try:
        code = child.wait(timeout=1900)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=10)
        raise RuntimeError('Paired workload deadline') from None
    emit('complete', child_returncode=code, optimizer_updates=0)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never stringify exceptions: network errors may contain signed capabilities.
        emit('failed', error_type=type(error).__name__)
        raise SystemExit(1) from None