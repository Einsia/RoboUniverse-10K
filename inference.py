"""Run the RoboUniverse-35B-A3B model on RoboUniverse-10K tasks with Claude Code in Harbor (Python 3.12).

download  fetch the weights from Hugging Face
serve     start SGLang with the settings the model was trained and evaluated with
check     verify a running server (text reply, tool calls, thinking off)
run       evaluate the model on one case or a split through `harbor run`
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from evaluate import materialize
from summarize import outcome

MODEL_REPO = 'Einsia/RoboUniverse-35B-A3B'
SERVED_NAME = 'RoboUniverse-35B-A3B'
CLAUDE_CODE_VERSION = '2.1.283'  # the version whose prompt and tools are in the training data
# Thinking stays off everywhere: no reasoning parser, the bundled chat template, and Claude Code with
# MAX_THINKING_TOKENS=0 (otherwise it sends a `thinking` field that SGLang rejects with HTTP 400).
SGLANG_FLAGS = ['--context-length', '262144', '--tool-call-parser', 'qwen3_coder', '--image-processor-backend', 'pil',
                '--moe-runner-backend', 'triton', '--kv-cache-dtype', 'bfloat16']
BLACKWELL_FLAGS = ['--attention-backend', 'trtllm_mha', '--page-size', '64']
# Claude Code settings of our evaluation harness: one call's output bound, the context window it assumes for the model,
# and auto-compaction sized to that window (compaction starts at 93% of 262144 - 8192 tokens).
CLAUDE_CODE_ENV = {
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '8192',
    'CLAUDE_CODE_MAX_CONTEXT_TOKENS': '262144',
    'CLAUDE_CODE_AUTO_COMPACT_WINDOW': '262144',
    'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '93',
    'DISABLE_COMPACT': '0',
    'DISABLE_AUTO_COMPACT': '0',
}
AGENT_TIMEOUT_MULTIPLIER = '2'  # our evaluation gave the agent 14,400 s; the tasks declare 7,200 s

def post(base, path, body, key='dummy', timeout=300):
    request = Request(base.rstrip('/') + path, json.dumps(body).encode(), method='POST', headers={
        'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': key})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, error.read().decode(errors='replace')

def served_models(base, timeout=10):
    with urlopen(base.rstrip('/') + '/v1/models', timeout=timeout) as response:
        return [m['id'] for m in json.loads(response.read())['data']]

def cmd_download(args):
    from huggingface_hub import snapshot_download
    print(snapshot_download(args.repo, revision=args.revision, local_dir=args.dir))

def cmd_serve(args):
    cmd = [args.python, '-m', 'sglang.launch_server', '--model-path', args.model, '--served-model-name', args.name,
           '--host', args.host, '--port', str(args.port), '--tp-size', str(args.tp),
           '--mem-fraction-static', str(args.mem_fraction), *SGLANG_FLAGS, *(BLACKWELL_FLAGS if args.blackwell else []),
           *(args.sglang_arg or [])]
    env = dict(os.environ)
    # SGLang compiles kernels with tools (ninja) installed next to its interpreter; works without activating the venv.
    env['PATH'] = os.pathsep.join([str(Path(args.python).absolute().parent), env.get('PATH', '')])
    if args.gpus: env['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('+', shlex.join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)

def cmd_check(args):
    ok = True
    def report(passed, message):
        nonlocal ok
        ok &= passed
        print(('PASS ' if passed else 'FAIL ') + message)
    try:
        models = served_models(args.server)
    except (URLError, OSError) as error:
        raise SystemExit(f'Cannot reach {args.server}/v1/models: {error}')
    report(args.name in models, f'served model names {models} include {args.name!r}')
    status, reply = post(args.server, '/v1/messages', {'model': args.name, 'max_tokens': 32, 'messages': [
        {'role': 'user', 'content': 'Reply with the single word ok.'}]}, args.api_key)
    text = ''.join(b.get('text', '') for b in reply.get('content', [])) if status == 200 else reply
    report(status == 200 and 'ok' in text.lower(), f'/v1/messages text reply: {status} {text!r}')
    status, reply = post(args.server, '/v1/messages', {'model': args.name, 'max_tokens': 256, 'tools': [{
        'name': 'Bash', 'description': 'Run a shell command and return its output.',
        'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}],
        'messages': [{'role': 'user', 'content': 'Use the Bash tool to run: echo hello'}]}, args.api_key)
    calls = [b for b in reply.get('content', []) if b.get('type') == 'tool_use'] if status == 200 else []
    report(bool(calls) and calls[0].get('name') == 'Bash',
           f'tool call parsed into a tool_use block: {calls[0]["input"] if calls else reply}'
           + ('' if calls else ' (start SGLang with --tool-call-parser qwen3_coder)'))
    status, reply = post(args.server, '/v1/messages', {'model': args.name, 'max_tokens': 16,
        'thinking': {'type': 'enabled', 'budget_tokens': 1024}, 'messages': [{'role': 'user', 'content': 'Hi'}]},
        args.api_key)
    report(status == 400, 'thinking is off on the server' if status == 400 else
           f'the server accepted a thinking request ({status}); it was probably started with --reasoning-parser, '
           'which turns thinking on. Restart it without that flag.')
    print('Clients must not request thinking: run Claude Code with MAX_THINKING_TOKENS=0 '
          '(`run` does this for Harbor).')
    raise SystemExit(0 if ok else 1)

def select(data, case, split):
    rows = {r['case']: r for r in csv.DictReader((data / 'catalog.csv').open())}
    if case:
        if case not in rows: raise SystemExit('Unknown case: ' + case)
        return [rows[case]]
    names = [r['case'] for r in csv.DictReader((data / f'splits/{split}.csv').open())] if split == 'heldout500' else rows
    return [rows[name] for name in names]

def cmd_run(args):
    host = urlparse(args.server).hostname
    if host in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
        raise SystemExit(f'--server {args.server} is not reachable from inside the agent container. '
                         "Use an address of the model server that Docker containers can reach, e.g. the host's IP.")
    try:
        models = served_models(args.server)
    except (URLError, OSError) as error:
        raise SystemExit(f'Cannot reach {args.server}/v1/models: {error}')
    if args.name not in models: raise SystemExit(f'{args.server} serves {models}, not {args.name!r}')
    if not args.accept_isaac_eula: raise SystemExit('Review the NVIDIA Isaac Sim EULA before passing --accept-isaac-eula')
    data, work = args.data.resolve(), args.work.resolve()
    selected = select(data, args.case, args.split)
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (work / 'selection.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=selected[0].keys()); writer.writeheader(); writer.writerows(selected)
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN')}
    no_proxy = ','.join(filter(None, [host, env.get('NO_PROXY') or env.get('no_proxy')]))
    env.update(ANTHROPIC_BASE_URL=args.server.rstrip('/'), ANTHROPIC_API_KEY=args.api_key, MAX_THINKING_TOKENS='0',
               NO_PROXY=no_proxy, no_proxy=no_proxy, ROBOENV_GPU_DEVICE=args.gpu, OMNI_KIT_ACCEPT_EULA='YES')
    results = Counter()
    for row in selected:
        dest = work / 'tasks' / row['case']
        materialize(data, row, dest, args.cache.resolve())
        cmd = ['harbor', 'run', '-p', str(dest), '-a', 'claude-code', '-m', f'anthropic/{args.name}',
               '--ak', f'version={args.claude_code_version}', '--ak', 'max_thinking_tokens=0',
               *[a for k, v in CLAUDE_CODE_ENV.items() for a in ('--ae', f'{k}={v}')],
               '--agent-timeout-multiplier', args.agent_timeout_multiplier,
               '-e', 'docker', '-n', '1', '--force-build', '--max-retries', '0',
               '--jobs-dir', str(work / 'jobs'), '--job-name', row['case']]
        print('+', shlex.join(cmd), flush=True)
        subprocess.run(cmd, env=env)
        result = outcome(work / 'jobs' / row['case']); results[result] += 1
        print(f'{row["case"]}: {result}', flush=True)
    scored = results['success'] + results['failure']
    print(json.dumps({'expected': len(selected), 'scored': scored, 'counts': results,
                      'success_rate_among_scored': results['success'] / scored if scored else None}, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download', help='download the weights from Hugging Face')
    p.add_argument('--repo', default=MODEL_REPO); p.add_argument('--revision'); p.add_argument('--dir', default='./' + SERVED_NAME)
    p.set_defaults(func=cmd_download)
    p = sub.add_parser('serve', help='start an SGLang server (run with the Python that has sglang installed)')
    p.add_argument('--model', default=MODEL_REPO, help='local directory or Hugging Face repo id')
    p.add_argument('--name', default=SERVED_NAME); p.add_argument('--host', default='0.0.0.0'); p.add_argument('--port', type=int, default=30000)
    p.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES for the server, e.g. 0 or 0,1')
    p.add_argument('--tp', type=int, default=1, help='tensor parallel size (2 on 80 GB GPUs)')
    p.add_argument('--mem-fraction', type=float, default=0.85)
    p.add_argument('--blackwell', action='store_true', help='add the faster attention backend we used on B200/B300')
    p.add_argument('--python', default=sys.executable, help='Python interpreter with sglang installed')
    p.add_argument('--sglang-arg', action='append', metavar='ARG',
                   help='extra argument passed to SGLang unchanged (repeatable), e.g. --sglang-arg=--log-requests')
    p.set_defaults(func=cmd_serve)
    for name, func in [('check', cmd_check), ('run', cmd_run)]:
        p = sub.add_parser(name, help=func.__doc__)
        p.add_argument('--server', required=True, help='model server URL, e.g. http://10.0.0.5:30000')
        p.add_argument('--name', default=SERVED_NAME, help='--served-model-name of the server')
        p.add_argument('--api-key', default='dummy')
        p.set_defaults(func=func)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--work', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--case'); group.add_argument('--split', choices=['heldout500', 'all'])
    p.add_argument('--gpu', default='0', help='GPU index or UUID for the simulator container')
    p.add_argument('--cache', type=Path, default=Path('.cache/robot-assets'))
    p.add_argument('--claude-code-version', default=CLAUDE_CODE_VERSION)
    p.add_argument('--agent-timeout-multiplier', default=AGENT_TIMEOUT_MULTIPLIER,
                   help='Harbor agent timeout multiplier (2 = the 14,400 s of our evaluation)')
    p.add_argument('--accept-isaac-eula', action='store_true')
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
