"""Report a single-pass success rate without silently dropping missing trials."""
import argparse, csv, json
from collections import Counter
from pathlib import Path

def outcome(job):
    trials=list(job.glob('*/result.json'))
    if not trials:return 'missing'
    if len(trials)!=1:return 'ambiguous'
    result=json.loads(trials[0].read_text())
    if result.get('exception_info') or any(s.get('exception_info') for s in result.get('step_results') or []):return 'error'
    rewards=(result.get('verifier_result') or {}).get('rewards') or {}
    value=rewards.get('reward')
    return {0:'failure',1:'success'}.get(value,'error')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    cases=[r['case'] for r in csv.DictReader(a.manifest.open())]
    if len(cases)!=len(set(cases)):raise ValueError('Duplicate cases in manifest')
    counts=Counter(outcome(a.jobs/case) for case in cases);n=counts['success']+counts['failure']
    print(json.dumps({'expected':len(cases),'scored':n,'counts':counts,'success_rate_among_scored':counts['success']/n if n else None},indent=2))
if __name__=='__main__':main()
