"""Shared paths, isolated environments and durable stage records for H20 runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading
import shutil

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/opt/hbv/trainir-h20-runtime')
MEG = RUNTIME/'Megatron-LM'
GALV = RUNTIME/'Hetu-Galvatron-dtsir'
DEFAULT_ROOT = REPO
PYTHONS = {s:Path('/opt/hbv/venv-'+s+'-h20/bin/python') for s in ('megatron','galvatron')}

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n')
    temp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment(system, out):
    out=Path(out)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(('DTSIR_','SLURM_')) or key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NCCL_IB_HCA','NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_NET','PYTORCH_CUDA_ALLOC_CONF'):
            env.pop(key)
    env.update(PYTHONPATH=str(MEG if system=='megatron' else GALV),
        PATH=str(PYTHONS[system].parent)+':/opt/hbv/venv-galvatron-h20/bin:/usr/local/cuda/bin:'+env['PATH'],
        CUDA_HOME='/usr/local/cuda',CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',
        AUTOMM='0',DTSIR_EXPERIMENT='off',MEGATRON_ROOT=str(MEG),GALV_REPO=str(GALV),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),
        TORCH_EXTENSIONS_DIR=str(out/'extensions'))
    if system=='megatron':
        env['CUDNN_HOME']='/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/cudnn'
        env['LD_LIBRARY_PATH']=env['CUDNN_HOME']+'/lib:'+env.get('LD_LIBRARY_PATH','')
    return env

def launch(system, argv, gpus=0):
    cmd=[str(PYTHONS[system]),'-u']
    if gpus:
        cmd+=['-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={gpus}']
    return cmd+list(map(str,argv))

def finish_search_timing(out,outcome='selected'):
    """End-to-end process-lifetime boundary, including imports and all preparation."""
    out=Path(out);target=out/'search_timing.json'
    if target.exists():return
    ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    hz=os.sysconf('SC_CLK_TCK');end=time.clock_gettime(time.CLOCK_BOOTTIME)
    save(target,dict(search_e2e_seconds=end-ticks/hz,method='Linux process start ticks to pre-training boundary',
         start_boot_ticks=ticks,clock_ticks_per_second=hz,end_boot_seconds=end,driver_pid=os.getpid(),
         outcome=outcome,scope='All workflow wall time from driver process start through ready-to-launch selected training. Includes imports, configuration, profile generation, compute/memory Profile, processing, process startup, native search and selected configuration validation/export. Communication calibration and selected training excluded.',
         end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def stage(out, name, cmd, cwd, env, timeout=3600):
    folder=Path(out)/'stages'; folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(name+'.json')
    if marker.exists(): raise RuntimeError(f'Stage already recorded: {marker}; inspect before replaying in a new directory')
    snapshot=folder/(name+'.code');snapshot.mkdir()
    for source in (REPO/'scripts').glob('*.py'): shutil.copy2(source,snapshot/source.name)
    if 'megatron' in str(cwd).lower():
        for filename in ('test_parallel_model.py','dtsir_collect.py'):
            shutil.copy2(Path(cwd)/filename,snapshot/filename)
    save(snapshot/'hashes.json',{f.name:sha(f) for f in snapshot.glob('*.py')})
    keys=['PYTHONPATH','CUDA_DEVICE_MAX_CONNECTIONS','OMP_NUM_THREADS','DTSIR_EXPERIMENT',
          'DTSIR_MML_LOGS','DTSIR_MEASURE_CANDIDATE_JSON','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR',
          'CUDNN_HOME','LD_LIBRARY_PATH','NUM_NODES','NUM_GPUS_PER_NODE']
    keys += ['PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF']
    keys += [k for k in env if k.startswith(('H20_', 'DTSIR_'))]
    record=dict(command=list(map(str,cmd)),cwd=str(cwd),environment={k:env[k] for k in keys if k in env},
                code_snapshot=str(snapshot),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',timeout_s=timeout)
    save(marker,record)
    if name=='selected_training' and (Path(out)/'contract.json').exists():finish_search_timing(out)
    begin=time.monotonic()
    with (folder/(name+'.log')).open('w') as f:
        process=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s',str(timeout)+'s']+record['command'],
                                 cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        record['pid']=process.pid; save(marker,record)
        stop=threading.Event()
        def monitor():
            with (folder/(name+'.gpu.csv')).open('w') as gpu:
                gpu.write('timestamp, index, uuid, memory.used [MiB], utilization.gpu [%], temperature.gpu, power.draw [W], clocks.current.sm [MHz], clocks.current.memory [MHz], power.limit [W], clocks_event_reasons.active\n')
                while not stop.is_set():
                    r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,power.limit,clocks_event_reasons.active','--format=csv,noheader'],capture_output=True,text=True)
                    gpu.write(r.stdout);gpu.flush();stop.wait(1)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try: code=process.wait()
        finally: stop.set();thread.join(timeout=10)
    record.update(returncode=code,seconds=time.monotonic()-begin,status='completed' if code==0 else 'timed_out' if code==124 else 'failed',
                  log_sha256=sha(folder/(name+'.log')))
    save(marker,record)
    if code: raise RuntimeError(f'Stage {name} {record["status"]}: {folder/(name+".log")}')
    return record
