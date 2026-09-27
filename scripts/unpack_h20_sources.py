#!/usr/bin/env python3
"""Verify and unpack the reviewed source snapshot into a NEW directory only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    review=json.loads((repo/'sources/review.json').read_text(encoding='utf-8'))
    archive=repo/review['source_archive']['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==review['source_archive']['sha256'], 'Archive hash mismatch'
    dest=Path(args.out).resolve()
    if dest.exists():
        raise RuntimeError('Use a new output directory; existing files are never overwritten')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        names=set()
        for member in members:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name:
                raise RuntimeError('Unsafe archive path')
            if not member.isfile() or member.size>100_000_000 or member.name.casefold() in names:
                raise RuntimeError('Unsafe archive member')
            names.add(member.name.casefold())
        dest.mkdir(parents=True,exist_ok=False)
        for member in members:
            target=dest.joinpath(*PurePosixPath(member.name).parts)
            target.resolve().relative_to(dest)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as out:
                out.write(src.read())
            target.chmod(member.mode & 0o777)
    manifest=json.loads((dest/'H20_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest:
        target=dest/row['path']
        if target.stat().st_size!=row['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Source hash mismatch: '+row['path'])
    print('SOURCE HASH CHECK PASS:',len(manifest),'files')
    print('Runtime source directory:',dest)
    print('No environment installation or GPU training has been run.')

if __name__=='__main__':
    main()
