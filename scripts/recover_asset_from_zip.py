#!/usr/bin/env python3
"""Recover one content-addressed training asset from an official ZIP member."""
from __future__ import annotations
import argparse, hashlib, os, tempfile, zipfile
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--zip',type=Path,required=True); p.add_argument('--member',required=True); p.add_argument('--sha256',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with zipfile.ZipFile(a.zip) as zf: payload=zf.read(a.member)
    digest=hashlib.sha256(payload).hexdigest()
    if digest != a.sha256: raise ValueError(f'{a.member}: expected {a.sha256}, got {digest}')
    if a.output.exists():
        existing=hashlib.sha256(a.output.read_bytes()).hexdigest()
        if existing == digest: print(f'reused {a.output}'); return
        raise FileExistsError(f'refusing to overwrite conflicting asset: {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=a.output.name+'.',dir=a.output.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(payload)
        os.replace(tmp,a.output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(f'recovered {a.output} sha256={digest}')
if __name__=='__main__': main()
