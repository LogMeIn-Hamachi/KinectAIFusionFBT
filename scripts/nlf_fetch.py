"""Fetch official NLF research assets, pin provenance, and verify transfer hashes."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/nlf/source'
OUT.mkdir(parents=True, exist_ok=True)

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'KinectFBT-development'}), timeout=60)

def getjson(url):
    with request(url) as r:
        return json.load(r)

records = []
for tag, name in [('v0.2.0', 'nlf_s_crop.pt'), ('v0.2.2', 'nlf_s_multi_0.2.2.torchscript')]:
    release = getjson('https://api.github.com/repos/isarandi/nlf/releases/tags/' + tag)
    asset = next(a for a in release['assets'] if a['name'] == name)
    path = OUT / name
    if not path.exists():
        part = path.with_suffix(path.suffix + '.partial')
        with request(asset['browser_download_url']) as src, part.open('wb') as dst:
            while chunk := src.read(8 << 20):
                dst.write(chunk)
        part.replace(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert path.stat().st_size == asset['size']
    if asset.get('digest'):
        assert asset['digest'] == 'sha256:' + digest
    records.append(dict(file=name, url=asset['browser_download_url'], sha256=digest, bytes=path.stat().st_size,
                        published=release['published_at'], publisher_digest=asset.get('digest')))
    print('Verified', name, digest, flush=True)

repo = ROOT / 'third_party/nlf'
if not repo.exists():
    commit = getjson('https://api.github.com/repos/isarandi/nlf/commits/main')['sha']
    with request('https://api.github.com/repos/isarandi/nlf/zipball/' + commit) as r:
        archive = r.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts[1:]
            if not parts or member.is_dir():
                continue
            target = repo.joinpath(*parts).resolve()
            assert target.is_relative_to(repo.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
    (repo / 'SOURCE-COMMIT.txt').write_text(commit + '\n')
else:
    commit = (repo / 'SOURCE-COMMIT.txt').read_text().strip()
(OUT / 'provenance.json').write_text(json.dumps(dict(repository='https://github.com/isarandi/nlf',
    commit=commit, license='Pretrained models: noncommercial research use only', assets=records), indent=2))
print('Source ready', commit, flush=True)
