import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
from common_space import candidates as common_candidates, require_equal, candidate_signature

root=Path('/opt/hbv/trainir-h20-handoff-received')
source=Path('/opt/hbv/trainir-h20-runtime/Megatron-LM/test_parallel_model.py').read_text()
tree=ast.parse(source)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_tpds_enumerate_structural_candidates')
space={'TPDS_RUNTIME':NS(config=NS(active=True,max_mbs=8))}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-enumerator>','exec'),space)
enumerate_actual=space[node.name]
cases=json.loads((root/'config/cases8.json').read_text())['cases']
rows=[]
for c in cases:
 model=NS(world_size=8,device_count=8,layer=c['num_layers'],head=c['num_attention_heads'],
          group=c['native_kv_heads'],seq=c['seq_length'],gbs=c['global_batch_size'],experts=1,mla=False)
 candidates=enumerate_actual(model)
 common=[s for s in candidates if s[2]==s[3]==1 and s[7] in (1,2,4,8)]
 assert len(common)==c['expected_structural_candidates'],(c['id'],len(common))
 common_rows=[(s[0],s[1],s[4],s[7]) for s in common]
 require_equal(common_rows,common_candidates(c),c['id'])
 assert any(s[3]>1 for s in candidates)
 for dp,pp,cp,up,tp,sp,ep,mbs,nmb in candidates:
  assert dp*pp*cp*up*tp==8 and sp==tp and ep==1
  assert model.gbs==dp*mbs*nmb and nmb>=pp
  assert model.head%(tp*up)==0 and model.group%(tp*up)==0
  assert model.seq%(2*cp*up)==0
 rows.append(dict(case=c['id'],common_count=len(common),expanded_native_kv_count=len(candidates),
                  common_candidate_sha256=candidate_signature(common_rows),
                  up_degrees=sorted({s[3] for s in candidates}),gqa_min=min(8,model.group)))
print(json.dumps(dict(passed=True,common_total=sum(r['common_count'] for r in rows),cases=rows),indent=2))
