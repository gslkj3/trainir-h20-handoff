"""Index recorded commands and immutable attempt evidence; never execute a replay."""
import json
from pathlib import Path
from h20_campaign import DEFAULT_ROOT, save, sha


def export(root=DEFAULT_ROOT):
    rows=[]
    for result in json.loads((root/'results.json').read_text()):
        attempts=[]
        folder=root/'runs'/result['case']/result['system']/result['space']
        for run in sorted(folder.glob('*')):
            if not run.is_dir():
                continue
            evidence_dirs=[run]+sorted(run.glob('training_retries/*'))
            stages=[]
            for directory in evidence_dirs:
                for path in sorted((directory/'stages').glob('*.json')):
                    stage=json.loads(path.read_text())
                    stages.append(dict(metadata=str(path),status=stage['status'],
                        command=stage['command'],cwd=stage['cwd'],environment=stage['environment'],
                        code_snapshot=stage.get('code_snapshot'),
                        log=str(path.with_suffix('.log')),log_sha256=stage.get('log_sha256'),
                        gpu_samples=str(path.with_suffix('.gpu.csv'))))
            files=[run/name for name in ('contract.json','space_definition.json',
                'selected_candidate.json','selected_runtime.yaml','search_timing.json')]
            files+=sorted((run/'selected').glob('*.json'))
            attempts.append(dict(run=str(run),artifacts=[dict(path=str(p),sha256=sha(p))
                for p in files if p.is_file()],stages=stages,
                training_summaries=[str(p/'training_summary.json') for p in evidence_dirs
                    if (p/'training_summary.json').exists()]))
        rows.append(dict(case=result['case'],system=result['system'],space=result['space'],
            status=result['status'],training_evidence=result.get('training_evidence'),attempts=attempts))
    save(root/'replay_index.json',dict(
        note='Recorded original commands and paths, not a script to execute in place. Use a fresh attempt or training retry directory. A replay using current code is distinct from restoring the recorded code snapshot; retain original evidence and dependencies. Running stages have incomplete logs and hashes.',
        rows=rows))
    print('Indexed replay rows:',len(rows),'stages:',sum(len(a['stages']) for r in rows for a in r['attempts']))


if __name__=='__main__':
    export()
