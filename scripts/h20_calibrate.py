"""Native H20 hardware calibration, excluded from compute/profile/search totals."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from h20_campaign import *

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
p.add_argument('--part',choices=['galvatron','nccl-build','nccl'],required=True); a=p.parse_args()
out=a.root/'hardware'; out.mkdir(exist_ok=True)
if a.part=='galvatron':
    hw=out/'galvatron'; hw.mkdir(exist_ok=False)
    (hw/'hardware_configs').mkdir()
    for file in (GALV/'galvatron/profile_hardware').glob('*.py'): shutil.copy2(file,hw/file.name)
    save(hw/'source_hashes.json',{f.name:sha(f) for f in hw.glob('*.py')})
    batches=list(map(str,[1024,512,256,128,64,32,16,8,4,2,1]))
    jobs=[('allreduce','profile_allreduce.py',['--global_tp_deg','8','4','2','--profile_time','0']),
          ('p2p','profile_p2p.py',['--pp_deg','2','4','8']),
          ('sp_allreduce','profile_allreduce.py',['--global_tp_deg','8','4','2','--profile_time','1','--local_batch_size']+batches),
          ('sp_all2all','profile_all2all.py',['--global_tp_deg','8','4','2','--local_batch_size']+batches),
          ('overlap','profile_overlap.py',['--overlap_time_multiply','4'])]
    for name,script,args in jobs:
        print('START native Galvatron hardware',name,flush=True)
        stage(hw,name,launch('galvatron',[hw/script]+args,8),hw,environment('galvatron',hw/name),1800)
    configs={f.name:json.loads(f.read_text()) for f in (hw/'hardware_configs').glob('*.json')}
    assert len(configs)==4,configs.keys()
    assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for d in configs.values() for v in d.values())
    save(hw/'summary.json',dict(status='completed',configs=configs,scope='Native hardware calibration; not search cost'))
elif a.part=='nccl-build':
    sdk=out/'nccl_sdk'; (sdk/'include').mkdir(parents=True); (sdk/'lib').mkdir()
    package=Path('/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/nccl')
    for f in (package/'include').glob('*.h'): (sdk/'include'/f.name).symlink_to(f)
    lib=package/'lib/libnccl.so.2'
    for name in ['libnccl.so','libnccl.so.2']: (sdk/'lib'/name).symlink_to(lib)
    cmd=['make','-C',str(a.root/'nccl-tests'),'-j8','MPI=0','CXX=g++','CUDA_HOME=/usr/local/cuda',
         'NCCL_HOME='+str(sdk),'BUILDDIR='+str(out/'nccl_build'),'NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90']
    stage(out,'nccl_build',cmd,a.root,environment('megatron',out/'nccl_compile'),1800)
    save(out/'nccl_build.json',dict(command=cmd,nccl_library=str(lib),nccl_library_sha256=sha(lib),
        nccl_tests_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.root/'nccl-tests',text=True).strip(),
        binaries={f.name:sha(f) for f in (out/'nccl_build').glob('*_perf')}))
else:
    ref=REPO/'reference/runtime_checks/a100_comm/a100_comm.py'
    spec=importlib.util.spec_from_file_location('native_comm_fit',ref); helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    os.environ['MEGATRON_ROOT']=str(MEG)
    raw=out/'devastator/comm_data'; raw.mkdir(parents=True,exist_ok=False)
    env=environment('megatron',raw)
    env['LD_LIBRARY_PATH']=str(out/'nccl_sdk/lib')+':'+env.get('LD_LIBRARY_PATH','')
    records=[]
    for op in helper.OPS:
        cmd=[str(out/'nccl_build'/(op+'_perf')),'-b','8','-e','1G','-f','2','-g','8','-d','float','-w','5','-n','20','-c','1']
        print('START nccl-tests',op,flush=True)
        r=stage(out/'devastator',op,cmd,a.root,env,600)
        shutil.copy2(out/'devastator/stages'/(op+'.log'),raw/(op+'_perf.txt'))
        points=helper.validate_raw((raw/(op+'_perf.txt')).read_text())
        records.append(dict(op=op,points=points,stage=r))
    processor=helper.processor_class()(str(raw),print_flag=True,force_reprocess=True)
    # Native processor reads the actual nccl-tests logs and saves its fitted model.
    model=json.loads((raw/'profile_comm.json').read_text())
    assert set(processor.result)==set(helper.OPS)
    for op in helper.OPS:
        parsed=processor.extract_data(str(raw/(op+'_perf.txt')))
        for size in parsed['data_size']:
            value=processor.data_predict([op,int(size)])
            assert math.isfinite(value) and value>0,(op,size,value)
    save(out/'devastator/summary.json',dict(status='completed',world_size=8,dtype='float32',records=records,model=model,
         scope='Native single communication model; not search cost; no imported historical measurements'))
