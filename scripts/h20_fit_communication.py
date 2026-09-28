"""Fit unchanged native model from validated logs; retain originals and format conversion."""
import importlib.util
import math
from h20_campaign import *
r=DEFAULT_ROOT;o=r/'hardware/devastator';raw=o/'comm_data'
spec=importlib.util.spec_from_file_location('fit',REPO/'reference/runtime_checks/a100_comm/a100_comm.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
os.environ['MEGATRON_ROOT']=str(MEG)
source=o/'stages/hypercube_threads8.log'
lines=[]
for line in source.read_text().splitlines():
 fields=line.split()
 if fields and fields[0].isdigit():
  assert len(fields)==12,fields
  fields.insert(3,'none');line=' '.join(fields)
 lines.append(line)
target=raw/'hypercube_perf.txt';target.write_text('\n'.join(lines)+'\n')
records=[]
for op in h.OPS:
 f=raw/(op+'_perf.txt');records.append(dict(op=op,points=h.validate_raw(f.read_text()),sha256=sha(f),threads=8 if op=='hypercube' else 1,gpus_per_thread=1 if op=='hypercube' else 8))
p=h.processor_class()(str(raw),print_flag=True,force_reprocess=True)
assert set(p.result)==set(h.OPS)
for op in h.OPS:
 for size in p.extract_data(str(raw/(op+'_perf.txt')))['data_size']:
  v=p.data_predict([op,int(size)]);assert math.isfinite(v) and v>0,(op,size,v)
save(o/'summary.json',dict(status='completed',world_size=8,records=records,model=json.loads((raw/'profile_comm.json').read_text()),retry_note='hypercube 1 thread x 8 GPUs failed correctness; unchanged binary 8 threads x 1 GPU passed. Original logs preserved; blank redop column represented as none in fitting input.',original_hypercube_sha256=sha(source),scope='Hardware calibration excluded from search cost'))
