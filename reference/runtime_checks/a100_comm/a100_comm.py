"""A100 communication-only job. Native Galvatron + native nccl-tests.

No training, synthetic timing, topology-aware model changes or old-cache overwrite.
Megatron's CommDataProcessor class is loaded verbatim by AST to avoid executing
the training module's unrelated import-time code. Its equations are unchanged.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
OPS = ('all_reduce', 'all_gather', 'reduce_scatter', 'sendrecv',
       'broadcast', 'reduce', 'alltoall', 'scatter', 'gather', 'hypercube')
BATCHES = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
NATIVE = ('profile_allreduce.py', 'profile_p2p.py', 'profile_all2all.py', 'profile_overlap.py')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def torch_nccl():
    import torch
    libraries = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libnccl.so' in line}
    if len(libraries) != 1:
        raise RuntimeError(f'Cannot identify the single NCCL library actually loaded by Torch: {libraries}')
    library = libraries.pop()
    value = ctypes.c_int()
    if ctypes.CDLL(str(library)).ncclGetVersion(ctypes.byref(value)) != 0:
        raise RuntimeError('ncclGetVersion failed')
    return library, value.value, torch.__version__


def processor_class():
    import glob
    import warnings
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    source = Path(os.environ['MEGATRON_ROOT']) / 'test_parallel_model.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CommDataProcessor']
    if len(nodes) != 1:
        raise RuntimeError('Expected one native CommDataProcessor')
    ns = dict(os=os, json=json, glob=glob, warnings=warnings, np=np, pd=pd,
              curve_fit=curve_fit, r2_score=r2_score)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns['CommDataProcessor']


def prepare(source):
    source = Path(source).resolve()
    if not (source / 'src/common.cu').is_file():
        raise RuntimeError(f'Not an nccl-tests source directory: {source}')
    library, nccl_version, torch_version = torch_nccl()
    previous = HERE / 'nccl_build.json'
    if previous.is_file():
        old = json.loads(previous.read_text())
        if (old.get('nccl_library') == str(library) and old.get('nccl_version') == nccl_version
                and old.get('source_common_sha256') == sha(source / 'src/common.cu')
                and all(Path(item['path']).is_file() and sha(item['path']) == item['sha256']
                        for item in old.get('binaries', {}).values())
                and set(old.get('binaries', {})) == set(OPS)):
            processor_class()
            print('REUSE verified nccl-tests build:', old['sdk'], flush=True)
            return
    root = Path(tempfile.mkdtemp(prefix='a100_nccl_build_', dir=os.environ['A100_WORK']))
    # Inspect known environment and compiler include paths, never the old repositories.
    candidates = [library.parent.parent / 'include/nccl.h', Path(sys.prefix) / 'include/nccl.h']
    for key in ('NCCL_HOME', 'NCCL_ROOT', 'NCCL_DIR'):
        if os.environ.get(key): candidates.append(Path(os.environ[key]) / 'include/nccl.h')
    for key in ('CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH'):
        candidates += [Path(p) / 'nccl.h' for p in os.environ.get(key, '').split(':') if p]
    candidates += [Path(p).parent / 'include/nccl.h'
                   for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    candidates += list(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/nccl/include/nccl.h'))
    header = next((p.resolve() for p in candidates if p.is_file()), None)
    if header is None:
        raise RuntimeError('Cannot locate nccl.h from the activated A100 modules/environment; no job submitted.')
    sdk = root / 'nccl_sdk'
    (sdk / 'include').mkdir(parents=True)
    (sdk / 'lib').mkdir()
    for item in header.parent.glob('*.h'):
        (sdk / 'include' / item.name).symlink_to(item.resolve())
    (sdk / 'lib/libnccl.so').symlink_to(library)
    (sdk / 'lib/libnccl.so.2').symlink_to(library)
    nvcc = shutil.which('nvcc')
    if not nvcc: raise RuntimeError('nvcc missing after loading A100 CUDA 12.1')
    cuda = Path(nvcc).resolve().parent.parent
    cmd = ['make', '-C', str(source), '-j4', 'MPI=0', 'CXX=g++',
           f'CUDA_HOME={cuda}', f'NCCL_HOME={sdk}', f'BUILDDIR={root / "build"}',
           'NVCC_GENCODE=-gencode=arch=compute_80,code=sm_80']
    print('BUILD', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(sdk / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    binaries = {}
    for op in OPS:
        exe = root / 'build' / (op + '_perf')
        if not exe.is_file(): raise RuntimeError(f'Missing {exe}')
        linked = subprocess.check_output(['ldd', str(exe)], env=env, text=True)
        if 'not found' in linked: raise RuntimeError(linked)
        (root / (op + '_ldd.txt')).write_text(linked)
        binaries[op] = dict(path=str(exe), sha256=sha(exe))
    record = dict(source=str(source), source_common_sha256=sha(source / 'src/common.cu'),
                  build_command=cmd, nccl_library=str(library), nccl_version=nccl_version,
                  torch=torch_version, header=str(header), sdk=str(sdk), binaries=binaries)
    save(root / 'build_info.json', record)
    save(HERE / 'nccl_build.json', record)
    processor_class()
    print('BUILD PASS', root, flush=True)


def preflight():
    record = json.loads((HERE / 'nccl_build.json').read_text())
    for item in record['binaries'].values():
        if not Path(item['path']).is_file() or sha(item['path']) != item['sha256']:
            raise RuntimeError('Prepared nccl-tests binary missing/changed; run prepare again')
    repo = Path(os.environ['GALV_REPO']).resolve()
    import galvatron
    if not Path(galvatron.__file__).resolve().is_relative_to(repo):
        raise RuntimeError('Galvatron imported from old repository')
    from galvatron.utils.training_utils import gen_profiling_groups  # noqa: F401
    for name in NATIVE:
        path = repo / 'galvatron/profile_hardware' / name
        ast.parse(path.read_text())
        result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f'{name} import/help failed:\n{result.stdout}\n{result.stderr}')
    print('PREFLIGHT PASS: native Galvatron entries and all NCCL binaries', flush=True)


def inventory(out):
    import torch
    out = Path(out)
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f'Expected 4 allocated visible GPUs, got {torch.cuda.device_count()}')
    gpu = [torch.cuda.get_device_name(i) for i in range(4)]
    if not all('A100' in n for n in gpu): raise RuntimeError(f'Wrong GPUs: {gpu}')
    library, version, torch_version = torch_nccl()
    system = 'galvatron' if 'galvatron-a100' in sys.prefix else 'megatron'
    folder = out / 'environment' / system
    folder.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    save(folder / (host + '.json'), dict(host=host, python=sys.executable,
         torch=torch_version, nccl_version=version, nccl_library=str(library), gpu=gpu,
         visible_mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
         environment={k:v for k,v in os.environ.items() if k.startswith(('NCCL_', 'GLOO_', 'SLURM_'))}))
    for label, argv in [('topology', ['nvidia-smi','topo','-m']), ('ip', ['ip','-br','addr']),
                        ('ib', ['ibdev2netdev'])]:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            text = result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc: text = str(exc)
        (folder / f'{host}_{label}.txt').write_text(text)


def validate_raw(text):
    if not re.search(r'Out of bounds values\s*:\s*0\s+OK', text):
        raise RuntimeError('Missing nccl-tests correctness PASS')
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 9 or not fields[0].isdigit(): continue
        size, timing, algbw = int(fields[0]), float(fields[5]), float(fields[6])
        if fields[8] not in ('N/A', '0'): raise RuntimeError('NCCL out-of-place correctness error')
        if len(fields) >= 13 and fields[12] not in ('N/A', '0'):
            raise RuntimeError('NCCL in-place correctness error')
        if size > 0 and timing > 0 and math.isfinite(timing) and math.isfinite(algbw):
            rows.append((size, timing, algbw))
    if len(rows) < 10: raise RuntimeError(f'Too few valid message sizes: {len(rows)}')
    return rows


def nccl(out):
    out = Path(out)
    record = json.loads((HERE / 'nccl_build.json').read_text())
    library, version, _ = torch_nccl()
    if str(library) != record['nccl_library'] or version != record['nccl_version']:
        raise RuntimeError('NCCL differs from preparation; rebuild before measurement')
    raw = out / 'megatron/comm_data'
    raw.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(Path(record['sdk']) / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    commands = []
    for op in OPS:
        exe = record['binaries'][op]['path']
        cmd = [exe, '-b', os.environ.get('A100_COMM_MIN', '8'),
               '-e', os.environ.get('A100_COMM_MAX', '1G'), '-f', '2', '-g', '4',
               '-d', os.environ.get('A100_COMM_DTYPE', 'float'), '-w', '5', '-n', '20', '-c', '1']
        print('NCCL START', op, flush=True)
        begin = time.monotonic()
        with (raw / (op + '_perf.txt')).open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        rows = validate_raw((raw / (op + '_perf.txt')).read_text())
        commands.append(dict(op=op, argv=cmd, rows=len(rows), seconds=time.monotonic()-begin))
        print('NCCL PASS', op, 'sizes=', len(rows), flush=True)
    processor = processor_class()(str(raw), print_flag=True, force_reprocess=True)
    if set(processor.result) != set(OPS): raise RuntimeError('Native communication model keys incomplete')
    for op, model in processor.result.items():
        if not all(math.isfinite(float(v)) for k,v in model.items() if k != 'model_type'):
            raise RuntimeError(f'Non-finite fit parameters: {op}')
        parsed = processor.extract_data(str(raw / (op + '_perf.txt')))
        if len(parsed) < 10: raise RuntimeError(f'Native parser did not accept raw data: {op}')
        for size in parsed['data_size']:
            prediction = processor.data_predict([op, int(size)])
            if not math.isfinite(prediction) or prediction <= 0:
                raise RuntimeError(f'Invalid native fitted time for {op}, size={size}')
    save(out / 'megatron/summary.json', dict(status='completed', nodes=1, gpus=4,
         commands=commands, build=record, processor_source_sha256=sha(Path(os.environ['MEGATRON_ROOT'])/'test_parallel_model.py'),
         model=str(raw/'profile_comm.json'), scope='Single platform curve per primitive; no topology/group dimension added.'))


def expected_config(stage):
    suffix = '4nodes_4gpus_per_node.json'
    if stage == 'allreduce':
        return 'allreduce_bandwidth_' + suffix, [f'allreduce_size_{tp}_consec_{c}'
            for tp in (16,8,4,2) for c in (1,0) if not (tp == 16 and c == 0)]
    if stage == 'p2p': return 'p2p_bandwidth_' + suffix, [f'pp_size_{p}' for p in (2,4,8,16)]
    if stage in ('sp_allreduce','sp_all2all'):
        op = 'allreduce' if stage == 'sp_allreduce' else 'all2all'
        return 'sp_time_' + suffix, [f'{op}_size_{t}_{b}MB_time' for t in (8,4,2) for b in BATCHES]
    return 'overlap_coefficient.json', ['overlap_coe']


def run(out):
    out = Path(out).resolve()
    if (out/'status.json').exists(): raise RuntimeError('Use a new submission/output directory')
    hw = out / 'galvatron/hardware'
    (hw/'hardware_configs').mkdir(parents=True)
    repo = Path(os.environ['GALV_REPO'])
    for file in (repo/'galvatron/profile_hardware').glob('*.py'):
        shutil.copy2(file, hw/file.name)
    save(out/'source_hashes.json', {name:sha(hw/name) for name in NATIVE})
    shutil.copy2(HERE/'nccl_build.json', out/'nccl_build.json')
    shutil.copy2(HERE/'a100_env.sh', out/'a100_env_used.sh')
    statuses = []

    def step(name, system, mode, nodes, argv=()):
        log = out/(name+'.log')
        cmd = ['srun','--exclusive','--kill-on-bad-exit=1','--time=00:30:00',
               f'--nodes={nodes}',f'--ntasks={nodes}','--ntasks-per-node=1',
               '--cpus-per-task=8','--gres=gpu:4','--unbuffered',
               'bash',str(HERE/'a100_comm_worker.sh'),system,mode,str(out),*argv]
        save(out/'status.json', dict(status='running', stage=name, completed_stages=statuses))
        print('START', name, 'log=', log, flush=True)
        begin = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(stage=name, returncode=result.returncode, seconds=time.monotonic()-begin, command=cmd)
        statuses.append(entry)
        if result.returncode: raise RuntimeError(f'{name} failed; see {log}')
        print('DONE', name, f'{entry["seconds"]:.2f}s', flush=True)

    failures = []
    try:
        step('galvatron_inventory','galvatron','inventory',4)
        if len(list((out/'environment/galvatron').glob('*.json'))) != 4:
            raise RuntimeError('Expected inventory from four distinct hosts')
        records = [json.loads(p.read_text()) for p in (out/'environment/galvatron').glob('*.json')]
        if len({(r['torch'], r['nccl_version']) for r in records}) != 1:
            raise RuntimeError('Torch/NCCL versions differ across allocated nodes')
        jobs = [
            ('allreduce','multi',4,['profile_allreduce.py','--global_tp_deg','16','8','4','2','--profile_time','0']),
            ('p2p','multi',4,['profile_p2p.py','--pp_deg','2','4','8','16']),
            ('sp_allreduce','multi',4,['profile_allreduce.py','--global_tp_deg','8','4','2','--local_batch_size',*map(str,BATCHES),'--profile_time','1']),
            ('sp_all2all','multi',4,['profile_all2all.py','--global_tp_deg','8','4','2','--local_batch_size',*map(str,BATCHES)]),
            ('overlap','local',1,['profile_overlap.py','--overlap_time_multiply','4'])]
        for name, mode, nodes, argv in jobs:
            step('galvatron_'+name, 'galvatron', mode, nodes, argv)
            filename, keys = expected_config(name)
            data = json.loads((hw/'hardware_configs'/filename).read_text())
            for key in keys:
                value = data.get(key)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f'Invalid/missing native Galvatron result {filename}: {key}={value}')
        save(out/'galvatron/summary.json', dict(status='completed', nodes=4, gpus_per_node=4,
             overlap_nodes=1, overlap_gpus=4, hardware_configs=str(hw/'hardware_configs')))
    except Exception as exc:
        failures.append('galvatron: '+str(exc))
        save(out/'galvatron/summary.json', dict(status='failed', error=str(exc)))
        print('GALVATRON FAILED:', exc, flush=True)
    # The independent Megatron measurement still runs if a Galvatron native
    # profiler fails. No fake/default coefficients are substituted.
    try:
        step('megatron_nccl4','megatron','nccl',1)
    except Exception as exc:
        failures.append('megatron: '+str(exc))
        save(out/'megatron/summary.json', dict(status='failed', error=str(exc)))
    save(out/'status.json', dict(status='failed' if failures else 'completed', failures=failures, stages=statuses))
    print('COMM RESULT:', out/'status.json', flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','preflight','inventory','nccl','run'])
    parser.add_argument('--source')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.source)
    elif args.action == 'preflight': preflight()
    elif args.action == 'inventory': inventory(args.out)
    elif args.action == 'nccl': nccl(args.out)
    else: run(args.out)
