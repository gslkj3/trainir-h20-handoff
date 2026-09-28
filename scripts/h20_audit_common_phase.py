"""Close the common phase only with audited outcomes and equal actual candidates."""
import json
from h20_campaign import DEFAULT_ROOT, save
from h20_workloads import cases
from common_space import candidates, require_equal, candidate_signature

root=DEFAULT_ROOT
success=json.loads((root/'completion_audit.json').read_text())['rows']
terminal=json.loads((root/'terminal_result_audit.json').read_text())['records']
verified={x.get('run'):x for x in success+terminal if x.get('verified')}
results={(x['case'],x['system'],x['space']):x for x in json.loads((root/'results.json').read_text())}
pairs=[]
for c in cases(root).values():
    record=dict(case=c['id'],verified=False,systems={},issues=[])
    signatures=[]
    try:
        for system in ('devastator','galvatron'):
            runs=sorted((root/'runs'/c['id']/system/'common').glob('*'))
            assert runs, f'No attempt: {system}'
            run=runs[-1]
            if system=='devastator':
                search=json.loads((run/'search.json').read_text())
                observed=[tuple(x['parallel'][i] for i in (0,1,4,7)) for x in search['evaluated']+search['rejected']]
            else:
                observed=[x['candidate'] for x in json.loads((run/'candidate_evaluations.json').read_text())]
            require_equal(observed,candidates(c),'actual common phase coverage')
            signature=candidate_signature(observed);signatures.append(signature)
            result=results[(c['id'],system,'common')]
            assert result['path']==str(run)
            state=result['status'] # Includes audited training-only retries.
            record['systems'][system]=dict(run=str(run),status=state,candidate_count=len(observed),candidate_sha256=signature)
            assert state in ('completed','no_feasible_candidate'),state
            assert str(run) in verified, f'Outcome evidence not verified: {system}'
        assert len(set(signatures))==1
        record['verified']=True
    except Exception as exc:
        record['issues'].append(repr(exc))
    pairs.append(record)
save(root/'common_phase_audit.json',dict(all_eight_pairs_verified=all(x['verified'] for x in pairs),
    verified_pairs=sum(x['verified'] for x in pairs),pairs=pairs,
    note='A native no-feasible result is distinct from successful training and from physical infeasibility. Full-space work is audited separately.'))
print('Verified common outcome pairs:',sum(x['verified'] for x in pairs),'/8')
