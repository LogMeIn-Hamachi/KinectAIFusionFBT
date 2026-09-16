"""Audit exactly the Git index, not ignored private working files. Prints paths, never secrets."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    paths=[p.decode('utf-8') for p in git('ls-files','-z').split(b'\0') if p]
    if not paths: raise RuntimeError('Stage the intended files before running the audit')
    failures=[];lfs=[]
    private_dirs={'recordings','diagnostics','cache','artifacts','release','build','third_party','.venv','__pycache__'}
    private_suffix={'.kfr','.mp4','.mkv','.avi','.png','.jpg','.jpeg','.bmp','.pdb','.log','.csv'}
    secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    local_path=re.compile(rb'(?i)[a-z]:[/\\](?:users|obs)[/\\][a-z0-9]')
    for path in paths:
        p=PurePosixPath(path)
        if set(p.parts)&private_dirs or p.suffix.lower() in private_suffix or p.name.startswith('calibration') and p.suffix=='.txt':
            failures.append((path,'private/output path'))
            continue
        data=git('show',':'+path)
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            m=re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n',data)
            if not m: failures.append((path,'malformed LFS pointer'));continue
            sha=m[1].decode();size=int(m[2]);file=ROOT/path
            if size>2*1024**3: failures.append((path,'over conservative 2 GiB LFS file limit'))
            with file.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=sha or file.stat().st_size!=size: failures.append((path,'LFS bytes differ from pointer'))
            lfs.append({'path':path,'bytes':size,'sha256':sha})
        else:
            if len(data)>20*1024**2: failures.append((path,'large binary outside LFS'))
            if secret.search(data): failures.append((path,'possible credential'))
            if local_path.search(data): failures.append((path,'personal absolute path'))
    report={'tracked_files':len(paths),'lfs_files':len(lfs),'lfs_bytes':sum(x['bytes'] for x in lfs),'findings':failures,'models':lfs}
    print(json.dumps(report,indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
