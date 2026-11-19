"""Bounded private input preparation; standard library, no simulator or credentials."""
import hashlib
from pathlib import Path
import time
from urllib.request import Request


class InputFailure(ValueError):
    """A public reason code, never an arbitrary exception string or signed URL."""
    def __init__(self, reason, **metrics):
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for chunk in iter(lambda: src.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_archive(config, work, opener, emit, validate_url, clock=time.monotonic):
    """Prefer a private mounted archive; remote fallback requires explicit opt-in.

    A CPU-side/public download is not proof of Flux-worker throughput. This helper
    does not provision a Flux mount or authorize another paid execution.
    """
    expected_bytes = config['input_bytes']
    expected_sha = config['input_sha256']
    if not isinstance(expected_bytes, int) or not 0 < expected_bytes <= 180_000_000:
        raise InputFailure('invalid_input_size')
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
        raise InputFailure('invalid_input_digest')
    mounted = config.get('input_archive')
    if mounted:
        archive = Path(mounted)
        if not archive.is_absolute() or archive.is_symlink() or not archive.is_file():
            raise InputFailure('local_input_missing')
        emit('input_source', mode='private_mount', expected_bytes=expected_bytes)
        mode = 'private_mount'
    else:
        if config.get('allow_remote_input') is not True:
            raise InputFailure('local_input_required')
        url = config.get('input_url')
        if not isinstance(url, str):
            raise InputFailure('remote_input_url_missing')
        validate_url(url, 270)
        archive = Path(work) / 'inputs.tar.gz'
        mode = 'scoped_https'
        emit('input_source', mode=mode, expected_bytes=expected_bytes, deadline_seconds=240)
        started = clock(); last_progress = started; size = 0
        try:
            with opener.open(Request(url, headers={'Accept-Encoding': 'identity'}), timeout=15) as response, archive.open('xb') as out:
                if response.status != 200:
                    raise InputFailure('input_http_status', http_status=int(response.status))
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != expected_bytes:
                    raise InputFailure('input_content_length_mismatch')
                while True:
                    chunk = response.read(65536)
                    now = clock(); elapsed = now - started
                    if elapsed > 240:
                        raise InputFailure('input_deadline_exceeded', received_bytes=size, elapsed_seconds=round(elapsed, 2))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected_bytes:
                        raise InputFailure('input_size_exceeded', received_bytes=size)
                    out.write(chunk)
                    if now - last_progress >= 5 or size == expected_bytes:
                        emit('input_progress', received_bytes=size, expected_bytes=expected_bytes,
                             elapsed_seconds=round(elapsed, 2), bytes_per_second=round(size / max(elapsed, 0.001), 2))
                        last_progress = now
                    # Stop predictably slow transfers, not a second paid network smoke.
                    if elapsed >= 30 and size < expected_bytes and size / elapsed * 240 < expected_bytes:
                        raise InputFailure('input_rate_insufficient', received_bytes=size,
                                           elapsed_seconds=round(elapsed, 2), expected_bytes=expected_bytes)
        except InputFailure:
            raise
        except (TimeoutError, OSError) as exc:
            raise InputFailure('input_transport_failure', transport_error_type=type(exc).__name__) from None
    actual_bytes = archive.stat().st_size
    if actual_bytes != expected_bytes:
        raise InputFailure('input_size_mismatch', received_bytes=actual_bytes, expected_bytes=expected_bytes)
    if digest(archive) != expected_sha:
        raise InputFailure('input_sha256_mismatch')
    emit('input_transfer_verified', mode=mode, received_bytes=actual_bytes, sha256=expected_sha)
    return archive