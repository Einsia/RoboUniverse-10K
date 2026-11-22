"""Download every unique upstream file without cookies, auth, netrc or local caches."""
import argparse, concurrent.futures, hashlib, json, os, tempfile, time, zipfile, csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit,quote
from urllib.request import build_opener, ProxyHandler, Request


def public_opener():
    proxies={}
    for scheme in ('https','http'):
        raw=os.environ.get(scheme.upper()+'_PROXY')
        if raw:
            p=urlsplit(raw)
            if p.username or p.password: raise ValueError('Authenticated proxies are excluded from anonymous checks')
            proxies[scheme]=raw
    return build_opener(ProxyHandler(proxies))

def verify(entry):
    url,expect=entry
    for attempt in range(3):
        try:
            request=Request(quote(url,safe=':/%'),headers={'Accept-Encoding':'identity'})
            with public_opener().open(request,timeout=45) as r:
                h=hashlib.sha256();size=0
                while chunk:=r.read(1<<20): h.update(chunk);size+=len(chunk)
                if r.status!=200 or h.hexdigest()!=expect:raise ValueError('HTTP or SHA256 mismatch')
                return {'url':url,'sha256':expect,'bytes':size,'http':r.status,'ok':True}
        except Exception as e:
            error=type(e).__name__+': '+str(e).split('?')[0][:160];time.sleep(attempt+1)
    return {'url':url,'sha256':expect,'ok':False,'error':error}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); cases={}
    for row in csv.DictReader((a.data/'catalog.csv').open()):
        with zipfile.ZipFile(a.data/row['archive']) as z:cases[row['case']]={tuple(v) for v in json.loads(z.read('upstream.json')).values()}
    refs=sorted(set().union(*cases.values()))
    with concurrent.futures.ThreadPoolExecutor(12) as pool: results=list(pool.map(verify,refs))
    failed={(r['url'],r['sha256']) for r in results if not r['ok']};blocked=[c for c,v in cases.items() if v & failed]
    summary={'utc':datetime.now(timezone.utc).isoformat(),'cases':len(cases),'checked':len(results),'failed':len(failed),'affected_cases':blocked,'results':results};a.output.write_text(json.dumps(summary,indent=2)+'\n');print({k:summary[k] for k in ('cases','checked','failed')},'affected',len(blocked));return bool(failed)
if __name__=='__main__':raise SystemExit(main())
