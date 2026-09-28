"""Verify supplied private archives and reuse existing identical input files.

Only manifest-listed payloads are installed. Never overwrites an existing file.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archives', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo/'sources/input_manifest.json').read_text())
    review = json.loads((repo/'sources/review.json').read_text())
    runtime = a.runtime.resolve(strict=True)
    if a.report.exists():
        raise RuntimeError('Report already exists')
    source_manifest = json.loads((runtime/'H20_SOURCE_MANIFEST.json').read_text())
    source_changed = [r['path'] for r in source_manifest if digest(runtime/r['path']) != r['sha256']]
    if source_changed:
        raise RuntimeError(f'Existing source differs from snapshot: {source_changed}')
    report = dict(source_files_verified=len(source_manifest), archives=[], files=[])
    with tempfile.TemporaryDirectory(prefix='h20-inputs-', dir=runtime.parent) as tmp:
        staging = Path(tmp)
        for filename, kind in [('h20_tokenizers_PRIVATE.tar.gz', 'tokenizers'),
                               ('a100_training_data.tar.gz', 'datasets')]:
            archive = a.archives/filename
            before = archive.stat()
            archive_hash = digest(archive)
            if kind == 'datasets' and archive_hash != review['dataset_archive_sha256']:
                raise RuntimeError('Dataset archive SHA256 mismatch')
            expected = {r['path']: r for r in manifest if r['kind'] == kind}
            with tarfile.open(archive, 'r:gz') as tar:
                seen = set()
                for member in tar.getmembers():
                    path = PurePosixPath(member.name)
                    if (not member.isfile() or path.is_absolute() or '..' in path.parts or
                        '\\' in member.name or ':' in member.name or member.name in seen):
                        raise RuntimeError('Unsafe or duplicate archive member')
                    seen.add(member.name)
                    if member.name == 'H20_INPUT_MANIFEST.json' and kind == 'tokenizers':
                        # The repository manifest remains the trust anchor.
                        if member.size > 1_000_000:
                            raise RuntimeError('Unexpected embedded manifest size')
                        continue
                    if member.name not in expected or member.size != expected[member.name]['bytes']:
                        raise RuntimeError('Unexpected member or size: ' + member.name)
                    target = staging/member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('xb') as dest:
                        shutil.copyfileobj(src, dest, 8 * 1024 * 1024)
                    if digest(target) != expected[member.name]['sha256']:
                        raise RuntimeError('Payload SHA256 mismatch: ' + member.name)
                if not set(expected).issubset(seen):
                    raise RuntimeError('Missing input members')
            after = archive.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Archive changed during validation')
            report['archives'].append(dict(name=filename, bytes=before.st_size,
                sha256=archive_hash, external_archive_hash_verified=kind == 'datasets'))
        # Check every existing destination before installing any new payload.
        for row in manifest:
            target = runtime/row['path']
            target.resolve().relative_to(runtime)
            if target.is_symlink() or (target.exists() and digest(target) != row['sha256']):
                raise RuntimeError('Refusing different existing input: ' + row['path'])
        for row in manifest:
            target = runtime/row['path']
            existed = target.exists()
            if not existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                with (staging/row['path']).open('rb') as src, target.open('xb') as dest:
                    shutil.copyfileobj(src, dest, 8 * 1024 * 1024)
            assert digest(target) == row['sha256']
            report['files'].append(dict(row, disposition='reused' if existed else 'installed'))
    report['passed'] = True
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print('PASS:', len(report['files']), 'private input files verified; source files preserved')


if __name__ == '__main__':
    main()
