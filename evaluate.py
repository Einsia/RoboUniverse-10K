"""Materialize a checked benchmark package and run Harbor locally (Python 3.12)."""
import argparse
import csv
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import zipfile
from summarize import outcome
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
# Minimum CPUs for the simulator container (the task packs ask for 4). Raise with ROBOUNIVERSE_SIM_CPUS on bigger hosts.
SIM_CPUS = float(os.environ.get("ROBOUNIVERSE_SIM_CPUS", "8"))

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def inside(root, name):
    path = Path(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Invalid relative path: {name}')
    return root / path

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def asset(item, cache, dest):
    name, (url, checksum) = item
    if not re.fullmatch('[0-9a-f]{64}', checksum) or not url.startswith((
            'https://raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/',
            'https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/',
        'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/')):
        raise ValueError(f'Unrecognized dependency for {name}')
    path = cache / checksum
    if not path.is_file() or digest(path) != checksum:
        for attempt in range(3):
            handle, name_tmp = tempfile.mkstemp(dir=cache)
            temp = Path(name_tmp)
            try:
                with os.fdopen(handle,'wb') as tmp, urlopen(quote(url,safe=':/%'), timeout=90) as src:
                    shutil.copyfileobj(src,tmp)
                if digest(temp) != checksum: raise ValueError(f'Upstream hash mismatch: {name}')
                temp.replace(path)
                break
            except (OSError,ValueError):
                if attempt == 2: raise
                time.sleep(attempt+1)
            finally: temp.unlink(missing_ok=True)
    target = inside(dest, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)

def materialize(data, row, dest, cache):
    source = inside(data, row['archive'])
    if digest(source) != row['sha256']: raise ValueError('Package checksum mismatch: '+row['case'])
    dest.mkdir(parents=True, exist_ok=False, mode=0o700)
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            path = inside(dest,name); path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as src, path.open('xb') as dst: shutil.copyfileobj(src,dst)
            if name.endswith('.sh') or path.name=='roboenv': path.chmod(0o755)
    if json.loads((dest/'release.json').read_text())['case'] != row['case']: raise ValueError('Case identity mismatch')
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda item:asset(item,cache,dest),json.loads((dest/'upstream.json').read_text()).items()))
    sim = dest/'environment/sim'; (sim/'runtime').mkdir(exist_ok=True)
    instance = json.loads((sim/'instance.json').read_text())
    tokens = {role: secrets.token_urlsafe(32) for role in ('agent','oracle')}
    instance['tokens']=tokens;write_json(sim/'instance.json',instance)
    for folder,role in [('environment','agent'),('solution','oracle')]:
        write_json(dest/folder/'connection.json',{'endpoint':'http://sim:8000','token':tokens[role]})
    for template,folder in [('agent','environment'),('sim','environment/sim'),('verifier','tests')]:
        shutil.copyfile(ROOT/'docker'/f'{template}.Dockerfile',dest/folder/'Dockerfile')
    compose = dest/'environment/docker-compose.yaml'
    deps=json.loads((sim/'workspace/dependencies.json').read_text())['runtime'].get('other_packages') or {}
    sidecar=sim/'bridge/embody_data/harbor/sidecar.py'
    source=sidecar.read_text()
    # NVIDIA's standalone image exposes VERSION, not an isaacsim wheel record.
    source=source.replace("importlib.metadata.version('isaacsim')", "Path('/isaac-sim/VERSION').read_text().strip().split('-')[0]")
    source=source.replace('if importlib.metadata.version(name) != version:', "if name.lower() == 'isaacsim' and same_release(Path('/isaac-sim/VERSION').read_text().strip().split('-')[0], version): continue\n        if importlib.metadata.version(name) != version:")
    sidecar.write_text(source)
    base={'numpy':'1.26.0','scipy':'1.15.3','pillow':'11.3.0','imageio':'2.37.0','imageio-ffmpeg':'0.8.1','isaacsim':'5.1.0.0','torch':'2.7.0+cu128'}
    extras=[]
    for name,version in deps.items():
        name=name.lower().replace('_','-')
        if not re.fullmatch(r'[a-z][a-z0-9-]*',name) or not re.fullmatch(r'[0-9][0-9a-z.+-]*',version):raise ValueError('Malformed runtime dependency')
        if base.get(name)!=version:extras.append(f'{name}=={version}')
    if extras:
        # Install only the task's extra packages. The base runtime (numpy, scipy, Pillow, ...) is pinned by the task's
        # dependencies.json and checked by the simulator sidecar at start, so pip must neither upgrade nor reinstall it:
        # `--upgrade` here pulled numpy 2.x in with matplotlib and the sidecar then refused to start.
        pins=' '.join(f'{name}=={version}' for name,version in sorted(base.items()) if name not in ('isaacsim','torch'))
        with (sim/'Dockerfile').open('a') as f:
            print("RUN printf '%s\\n' "+pins+" > /tmp/roboenv-constraints.txt && /isaac-sim/python.sh -m pip install --no-cache-dir --target /opt/roboenv/extra -c /tmp/roboenv-constraints.txt "+' '.join(sorted(extras)),file=f)
    # Some task workspaces read importlib.metadata.version('isaacsim') themselves (e.g. 001918_lid_cover). The platform's
    # simulator image installs Isaac Sim from pip wheels and has that metadata; NVIDIA's container used here does not, so
    # those tasks failed at start with PackageNotFoundError. Add metadata only (no code) for the Isaac Sim the image runs.
    with (sim/'Dockerfile').open('a') as f:
        print("RUN d=/opt/roboenv/extra/isaacsim-5.1.0.0.dist-info && mkdir -p $d && "
              "printf 'Metadata-Version: 2.1\\nName: isaacsim\\nVersion: 5.1.0.0\\n' > $d/METADATA && : > $d/RECORD", file=f)
    cfg = json.loads(compose.read_text())
    services = cfg['services']
    services['sim']['environment']['OMNI_KIT_ACCEPT_EULA'] = '${OMNI_KIT_ACCEPT_EULA:?review NVIDIA EULA and set YES}'
    # The task's own compose gives the simulator 4 CPUs. A cold Isaac Sim start (shader and scene setup) then takes
    # more than the task's own 180-300 s readiness wait on many tasks, which fails the trial before the agent starts.
    # Measured on one RTX PRO 5000 host: 4 CPUs never ready in 244 s, 8 CPUs ready in <=117 s, 16 CPUs in <=76 s.
    services['sim']['cpus'] = max(float(services['sim'].get('cpus') or 0), SIM_CPUS)
    # Harbor uses the task's environment directory as the Compose project directory.
    # Keep both ends of the per-episode connection outside the image build cache.
    for service, context, source in [
        ('main', compose.parent, './connection.json'),
        ('sim', sim, './sim/instance.json'),
    ]:
        filename = Path(source).name
        services[service].setdefault('volumes', []).append({
            'type': 'bind',
            'source': source,
            'target': f'/opt/roboenv/{filename}',
            'read_only': True,
            'bind': {'create_host_path': False},
        })
        with (context / '.dockerignore').open('a') as ignore:
            ignore.write(f'\n/{filename}\n')
    write_json(compose, cfg)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['list','prepare','run']);parser.add_argument('--data',type=Path,required=True)
    select=parser.add_mutually_exclusive_group();select.add_argument('--case');select.add_argument('--split',choices=['heldout500','all'])
    parser.add_argument('--work',type=Path);parser.add_argument('--cache',type=Path,default=Path('.cache/robot-assets'))
    parser.add_argument('--agent',default='oracle');parser.add_argument('--model');parser.add_argument('--gpu',default='0');parser.add_argument('--accept-isaac-eula',action='store_true')
    args=parser.parse_args();data=args.data.resolve();rows={r['case']:r for r in csv.DictReader((data/'catalog.csv').open())}
    if args.case and args.case not in rows:parser.error('Unknown case: '+args.case)
    if args.split=='heldout500':
        selected=[rows[r['case']] for r in csv.DictReader((data/'splits/heldout500.csv').open())]
    elif args.case:selected=[rows[args.case]]
    elif args.split=='all' or args.command=='list':selected=list(rows.values())
    else:parser.error('Choose --case or --split explicitly')
    if args.command=='list':
        for r in selected:print(r['case'],r['robot']);
        return
    if not args.work:parser.error('A fresh --work directory is required')
    if args.command=='run' and not args.accept_isaac_eula:parser.error('Review the NVIDIA Isaac Sim EULA before passing --accept-isaac-eula')
    work=args.work.resolve();work.mkdir(mode=0o700,parents=True,exist_ok=False)
    with (work/'selection.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=selected[0].keys());w.writeheader();w.writerows(selected)
    infrastructure_errors=[]
    for row in selected:
        dest=work/'tasks'/row['case'];materialize(data,row,dest,args.cache.resolve())
        if args.command=='run':
            cmd=['harbor','run','-p',str(dest),'-a',args.agent,'-e','docker','-n','1','--force-build','--max-retries','0','--jobs-dir',str(work/'jobs'),'--job-name',row['case']]
            if args.model:cmd+=['-m',args.model]
            subprocess.run(cmd,check=True,env={**os.environ,'ROBOENV_GPU_DEVICE':args.gpu,'OMNI_KIT_ACCEPT_EULA':'YES'})
            if outcome(work/'jobs'/row['case']) not in ('success','failure'):
                infrastructure_errors.append(row['case'])
    if infrastructure_errors:
        raise SystemExit(f'{len(infrastructure_errors)} unscored trial(s). Inspect the job results.')

if __name__=='__main__':main()
