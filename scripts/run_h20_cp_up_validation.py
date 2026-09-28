import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import argparse

p=argparse.ArgumentParser()
p.add_argument('kind', choices=['cp2','up2','cp2-up2','up2-dp2'])
p.add_argument('--out', type=Path, help='New output directory; existing results are never overwritten.')
a=p.parse_args()
kind=a.kind
cp,up,tp,dp={'cp2':(2,1,2,1),'up2':(1,2,2,1),
             'cp2-up2':(2,2,1,1),'up2-dp2':(1,2,1,2)}[kind]
root=Path('/opt/hbv/trainir-h20-cp-up-validation-20260927')
out=a.out.resolve() if a.out else root/(kind+'-01')
out.mkdir(parents=True,exist_ok=False)
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not busy, 'GPU compute processes already present: '+busy
base=Path('/opt/hbv/trainir-h20-validation-20260927/megatron-llama3-tp4-bf16-01/launch.json')
cmd=json.loads(base.read_text())['command']
candidate=dict(candidate_id=kind,parallel=[dp,2,cp,up,tp,tp,1,1 if dp==2 else 2,4],
 strategy={'ReCompute':[None,None],'VirtualPipe':None,'DistributedOptimizer':False,'Hybrid_MHA_MQA':[True,4]})
cmd[cmd.index('--data-cache-path')+1]=str(out/'dataset-cache')
env=dict(os.environ)
for key in list(env):
 if key.startswith('DTSIR_') or key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
  env.pop(key)
env.update(PYTHONPATH='/opt/hbv/trainir-h20-runtime/Megatron-LM',
 PATH='/opt/hbv/venv-megatron-h20/bin:/opt/hbv/venv-galvatron-h20/bin:/usr/local/cuda/bin:'+env['PATH'],
 CUDA_HOME='/usr/local/cuda',CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',AUTOMM='0',DTSIR_EXPERIMENT='off',
 DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(candidate),H20_VALIDATION_OUT=str(out),
 MEGATRON_ROOT='/opt/hbv/trainir-h20-runtime/Megatron-LM',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 CUDNN_HOME='/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/cudnn',
 LD_LIBRARY_PATH='/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/cudnn/lib:'+env.get('LD_LIBRARY_PATH',''),
 TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),TORCH_EXTENSIONS_DIR=str(out/'extensions'))
(out/'launch.json').write_text(json.dumps(dict(command=cmd,candidate=candidate,scope='Native small-model CP/UP integration, not performance or full-search validation.'),indent=2))
with (out/'train.log').open('w') as f:
 result=subprocess.run(['timeout','--kill-after=30s','900s']+cmd,env=env,cwd=env['MEGATRON_ROOT'],stdout=f,stderr=subprocess.STDOUT)
ranks=[json.loads(p.read_text()) for p in out.glob('rank[0-7].json')]
log=(out/'train.log').read_text(errors='replace')
losses=[float(x) for x in re.findall(r'lm loss:\s*([^ |]+)',log)]
skips=re.findall(r'number of skipped iterations:\s*(\d+)',log)
comm='a2a+p2p' if cp>1 and up>1 else 'a2a' if up>1 else 'p2p'
passed=result.returncode==0 and len(ranks)==8 and len(losses)==10 and all(math.isfinite(x) for x in losses)
passed=passed and len(skips)==10 and all(int(x)==0 for x in skips)
passed=passed and {r['rank'] for r in ranks}==set(range(8)) and {r['device'] for r in ranks}==set(range(8))
passed=passed and all(r['passed'] and len(r['updates'])==10 and r['effective']['context_parallel_size']==cp*up
 and r['effective']['cp_comm_type']==[comm] and r['effective']['tensor_model_parallel_size']==tp
 and r['effective']['data_parallel_size']==dp for r in ranks)
status=dict(passed=passed,exit_code=result.returncode,ranks=len(ranks),cp=cp,up=up,tp=tp,pp=2,dp=dp,communication=comm,losses=losses)
(out/'status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
sys.exit(0 if passed else 1)
