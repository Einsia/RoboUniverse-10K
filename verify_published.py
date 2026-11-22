"""After publication: verify ungated access and every ZIP at an immutable HF commit."""
import argparse, csv, hashlib, io, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from verify_anonymous import public_opener, verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--revision',required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'[\w.-]+/[\w.-]+',a.repo) or not re.fullmatch('[0-9a-f]{40}',a.revision):p.error('Expected owner/name and full 40-character HF commit')
    api='https://huggingface.co/api/datasets/'+a.repo+'/revision/'+a.revision
    with public_opener().open(api,timeout=30) as r:meta=json.load(r)
    if meta.get('private') is not False or meta.get('gated') not in (False,None) or meta.get('sha')!=a.revision:raise SystemExit('HF dataset is private, gated, or not the selected commit')
    base=f'https://huggingface.co/datasets/{a.repo}/resolve/{a.revision}/'
    with public_opener().open(base+'catalog.csv',timeout=30) as r:remote=r.read()
    if remote!=a.catalog.read_bytes():raise SystemExit('Published catalog differs from local release')
    rows=list(csv.DictReader(io.StringIO(remote.decode())))
    with ThreadPoolExecutor(8) as ex: results=list(ex.map(verify,[(base+r['archive'],r['sha256']) for r in rows]))
    failed=sum(not x['ok'] for x in results);a.output.write_text(json.dumps(dict(repo=a.repo,revision=a.revision,checked=len(results),failed=failed,results=results),indent=2));print('checked',len(results),'failed',failed);raise SystemExit(bool(failed))
