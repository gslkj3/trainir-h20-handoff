"""Five shared-space workflows using Galvatron's native profiler/cost model.

Pinned source: cea12ffb146a220643c8f99f9cb84294755d29f8.
This is an integration harness, not a claim of completed hardware validation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REVISION = 'cea12ffb146a220643c8f99f9cb84294755d29f8'


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n')
    temp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def yaml_write(path, data):
    # JSON is also valid YAML; Hydra accepts the .yaml filename.
    write(path, data)


def model_fields(c):
    family = 'qwen' if c['id'].startswith('qwen') else 'llama'
    return dict(model_size=family+'-'+c['id'], hidden_size=c['hidden'],
                ffn_hidden_size=c['ffn'], num_layers=c['layers'],
                num_attention_heads=c['heads'], num_query_groups=c['kv'],
                kv_channels=c['hidden']//c['heads'], vocab_size=c['vocab'],
                padded_vocab_size=c['vocab'], normalization='RMSNorm',
                norm_epsilon=c['eps'], layernorm_epsilon=c['eps'],
                activation_func='torch.nn.functional.silu', gated_linear_unit=True,
                position_embedding_type='rope', rotary_base=c['rope'],
                apply_rope_fusion=False, add_bias_linear=False,
                add_qkv_bias=c['qkv_bias'], qk_layernorm=c['qk_norm'],
                untie_embeddings_and_output_weights=c['untied'],
                make_vocab_size_divisible_by=1, initialize_on_meta=1,
                print_loss=1, dropout_prob=0.0)


def runtime_config(c, megatron):
    from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
    r = GalvatronRuntimeArgs().model_dump(exclude={'model': {'params_dtype'}})
    r['model'].update(model_fields(c))
    r['parallel'].update(pp_deg=1, global_tp_deg=1, global_cp_deg=1,
                         global_ep_deg=1, global_checkpoint=0, sdp=0,
                         vocab_tp=1, vocab_sdp=0, mixed_precision=c['dtype'],
                         default_dp_type='ddp', pipeline_type='pipedream_flush',
                         use_ulysses=False)
    r['train'].update(train_iters=10, eval_iters=0, iteration=0,
                      global_batch_size=c['gbs'], micro_batch_size=1,
                      chunks=1, seq_length=c['seq'], sequence_parallel=True,
                      use_flash_attn=True,
                      lr=1.25e-6 if c['id'] in ('llama3_8b_8k','qwen3_14b_4k') else 1e-6,
                      min_lr=1e-7,
                      lr_warmup_fraction=0.01, init_method_std=0.01, weight_decay=0.1,
                      adam_beta1=0.9, adam_beta2=0.95, num_workers=0, seed=42)
    r['data'].update(use_random_dataset=False,
                     data_path=[str(megatron/c['data'])], split='10,0,0',
                     tokenizer_type='HuggingFaceTokenizer',
                     tokenizer_model=str(megatron/c['tokenizer']))
    r['profile'].update(profile=0, exit_after_profiling=0)
    r['ckpt'].update(load=None, save=None)
    r['distributed_timeout_minutes'] = 15
    return r


def qwen3_adapter():
    """Use the backend's existing RMSNorm for Q/K, not nn.LayerNorm.

    Scope: constructor binding only, in this process; no upstream file edits.
    The selected revision passes incompatible nn.LayerNorm constructor kwargs.
    """
    from galvatron.core.runtime.models import modules
    from galvatron.core.runtime.transformer.norm import GalvatronNorm
    native = modules.SelfAttention
    class Qwen3SelfAttention(native):
        def __init__(self, config, submodules, *args, **kwargs):
            if config.qk_layernorm:
                submodules = copy.copy(submodules)
                submodules.q_layernorm = GalvatronNorm
                submodules.k_layernorm = GalvatronNorm
            super().__init__(config, submodules, *args, **kwargs)
    modules.SelfAttention = Qwen3SelfAttention


def stage(out, name, argv, cwd, nodes=0, gpus=0, extra_env=None):
    """A success marker requires exit zero; final training also requires summary.
    Native raw-profile processing is separately checked before search.
    """
    spec = dict(name=name, argv=list(map(str, argv)), cwd=str(cwd),
                nodes=nodes, gpus=gpus, env=extra_env or {})
    signature = digest(spec)
    marker = out/'stages'/f'{name}.done.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['signature'] != signature:
            raise RuntimeError(f'Changed completed stage {name}; choose a new GALV6_TAG')
        print('SKIP', name, flush=True)
        return old['seconds']
    write(out/'status.json', dict(status='running', stage=name))
    print('START', name, flush=True)
    log = out/'stages'/f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **(extra_env or {}))
    if nodes:
        specfile = out/'stages'/f'{name}.launch.json'
        write(specfile, spec)
        command = ['srun', '--exclusive', '--kill-on-bad-exit=1',
                   '--time='+os.environ.get('GALV6_STAGE_LIMIT', '02:00:00'),
                   f'--nodes={nodes}', f'--ntasks={nodes}', '--ntasks-per-node=1',
                   '--cpus-per-task=8', f'--gres=gpu:{gpus}',
                   'bash', str(HERE/'worker.sh'), str(specfile)]
    else:
        command = spec['argv']
    begin = time.perf_counter()
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - begin
    if result.returncode:
        write(out/'status.json', dict(status='failed', stage=name,
                                      returncode=result.returncode, log=str(log)))
        raise RuntimeError(f'{name} failed ({result.returncode}); see {log}')
    write(marker, dict(signature=signature, seconds=seconds, spec=spec))
    print('DONE', name, f'{seconds:.2f}s', flush=True)
    return seconds


def worker(specfile):
    spec = json.loads(Path(specfile).read_text())
    os.chdir(spec['cwd'])
    os.environ.update(spec['env'])
    # SLURM_PROCID is the rank in THIS srun step, not the array task index.
    rank = int(os.environ['SLURM_PROCID'])
    nodes, gpus = spec['nodes'], spec['gpus']
    if nodes == 1:
        os.environ.update(NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='lo',
                          GLOO_SOCKET_IFNAME='lo')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run',
               f'--nnodes={nodes}', f'--nproc-per-node={gpus}', f'--node-rank={rank}']
    if nodes == 1:
        command += ['--standalone']
    else:
        command += [f"--master-addr={os.environ['MASTER_ADDR']}",
                    f"--master-port={os.environ['MASTER_PORT']}"]
    command += spec['argv']
    print('LAUNCH', shlex.join(command), flush=True)
    raise SystemExit(subprocess.call(command))


def train_entry(config, overrides, source, output, qk=False):
    # Cache-only compatibility fix: execute in each NEW torchrun child,
    # before importing torch or upstream training code. Never delete live caches.
    import socket
    import tempfile
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    job = os.environ.get('SLURM_JOB_ID', 'nojob')
    host = socket.gethostname()
    cache_parent = Path('/tmp') / f'galv6-cache-v2-{os.getuid()}-{job}-{host}'
    cache_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A distinct directory per process also prevents stale metadata from a prior
    # profiling launch. It does not change kernel selection or disable compile.
    cache = Path(tempfile.mkdtemp(prefix=f'rank{rank}-', dir=cache_parent))
    previous = {k: os.environ.get(k) for k in
                ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')}
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
    os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    for key in previous:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    event = dict(policy='node-local-per-process-v2', host=host, rank=rank,
                 pid=os.getpid(), job=job, previous=previous,
                 current={k: os.environ[k] for k in previous},
                 config=str(config), timestamp=time.time())
    event_path = Path(config).resolve().parent / 'cache_policy_events' / (
        f'{host}-rank{rank}-pid{os.getpid()}.json')
    write(event_path, event)
    print('GALV6_CACHE_POLICY ' + json.dumps(event), flush=True)
    import torch
    import torch.distributed as dist
    from galvatron.core.arguments import load_with_hydra
    if qk:
        qwen3_adapter()
    # Load the upstream entry as a module, preserving its training loop.
    m = runpy.run_path(source, run_name='galvatron_six_native')
    args = load_with_hydra(config, overrides=overrides, mode='train_dist')
    from galvatron.utils.hf_config_adapter import resolve_model_config
    resolve_model_config(args)
    rows = []
    if output:
        class Timer:
            def profile_memory(self, *a, **kw): pass
            def post_profile_memory(self, *a, **kw): pass
            def profile_time_start(self, iteration):
                torch.cuda.synchronize()
                self.start = time.perf_counter()
            def profile_time_end(self, iteration, loss=None, learning_rate=None, grad_norm=None):
                torch.cuda.synchronize()
                def scalar(x):
                    return None if x is None else float(x.item() if hasattr(x, 'item') else x)
                row = dict(iteration=iteration+1, seconds=time.perf_counter()-self.start,
                           loss=scalar(loss), grad_norm=scalar(grad_norm))
                rows.append(row)
                print('GALV6_STEP', json.dumps(dict(rank=dist.get_rank(), **row)), flush=True)
        m['train'].__globals__['get_runtime_profiler'] = lambda *a, **kw: Timer()
    m['initialize_galvatron'](args)
    try:
        m['train'](args)
        if output:
            rank, world = dist.get_rank(), dist.get_world_size()
            out = Path(output)
            write(out/f'rank{rank}_iterations.json', rows)
            write(out/f'rank{rank}_effective.json', args.model_dump())
            allrows = [None]*world
            dist.all_gather_object(allrows, rows)
            assert world == 16
            assert all([r['iteration'] for r in rs] == list(range(1, 11)) for rs in allrows)
            for rs in allrows:
                for row in rs:
                    assert math.isfinite(row['seconds']) and row['seconds'] > 0
                    assert all(row[k] is None or math.isfinite(row[k]) for k in ('loss', 'grad_norm'))
            assert all(any(rs[i]['loss'] is not None for rs in allrows) for i in range(10))
            times = [max(rs[i]['seconds'] for rs in allrows) for i in range(10)]
            mean = statistics.mean(times[5:])
            if rank == 0:
                write(out/'training_summary.json', dict(status='completed', iterations=10,
                      independent_runs=1, measurement_iterations=[6,7,8,9,10],
                      mean_last5_iteration_s=mean, all_rank_max_iteration_s=times,
                      global_batch_size=args.train.global_batch_size,
                      samples_per_second=args.train.global_batch_size/mean,
                      timing_scope='Forward/backward, optimizer and zero_grad; excludes batch loading and final barrier',
                      sequence_parallel=args.train.sequence_parallel,
                      qwen3_existing_rmsnorm_adapter=qk))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def preflight(c, r, repo, out):
    import torch
    from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
    from galvatron.core.runtime.datasets.megatron.tokenizer import build_tokenizer
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert revision == REVISION, f'Unsupported source revision: {revision}'
    assert torch.cuda.device_count() == 8, 'Expected eight visible GPUs on batch node'
    assert all('5090' in torch.cuda.get_device_name(i) for i in range(8))
    for suffix in ('.bin', '.idx'):
        assert Path(r['data']['data_path'][0]+suffix).is_file(), r['data']['data_path']
    a = GalvatronRuntimeArgs.model_validate(r)
    tok = build_tokenizer(a, local_files_only=True)
    assert 0 < tok.vocab_size <= c['vocab'], (tok.vocab_size, c['vocab'])
    if c['qk_norm']:
        # Test the same existing RMSNorm used by the scoped Q/K adapter.
        from galvatron.core.runtime.transformer.norm import GalvatronNorm
        norm = GalvatronNorm(a.model, c['hidden']//c['heads'], eps=c['eps'])
        x = torch.randn(2, 3, c['hidden']//c['heads'], device='cuda', requires_grad=True)
        y = norm(x)
        ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True)+c['eps']) * norm.weight
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-5)
        grad = torch.randn_like(y)
        g1 = torch.autograd.grad(y, (x,norm.weight), grad, retain_graph=True)
        g2 = torch.autograd.grad(ref, (x,norm.weight), grad)
        for actual, expected in zip(g1,g2):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        write(out/'qk_rmsnorm_check.json', dict(passed=True,
              scope='Existing backend RMSNorm local output/gradient, not a full-model equivalence test'))
    # Pin actual versions rather than infer them from module names.
    record = dict(revision=revision, torch=torch.__version__, cuda=torch.version.cuda,
                  nccl=torch.cuda.nccl.version(), tokenizer_vocab=tok.vocab_size,
                  configured_vocab=c['vocab'], runtime=r,
                  source_changes=subprocess.check_output(['git','diff','--stat'], cwd=repo, text=True),
                  interpretation='Native Galvatron DP/TP/PP-restricted workflow; shared-space equality requires a separate contract comparison')
    write(out/'environment.json', record)


def common_rows(c):
    from common5_space import candidates
    return candidates(layers=c['layers'], hidden=c['hidden'], heads=c['heads'],
                      kv_heads=c['kv'], global_batch=c['gbs'])


def prepare_profiles(c, r, out, gpt, kind):
    """Generate and validate native commands without starting a GPU process."""
    from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs
    from galvatron.core.profiler.model_profiler import ModelProfiler
    pr = copy.deepcopy(r)
    model_template = out/'model_template.yaml'
    yaml_write(model_template, {})
    pr['model']['model_config_path'] = str(model_template)
    pr['train'].update(train_iters=20, chunks=1)
    pr['model'].update(set_layernum_manually=1, set_seqlen_manually=1, initialize_on_meta=0)
    pr['parallel'].update(async_grad_reduce=False)
    pr['profile'].update(profile=1, exit_after_profiling=1)
    pr['data']['use_random_dataset'] = True
    template = out/f'{kind}_runtime.yaml'
    yaml_write(template, {'runtime':pr})
    # This is GLOBAL batch size. DP16 calibration needs at least 16, not 4.
    pa = GalvatronModelProfilerArgs(profile_type=kind, profile_mode='static',
         profile_unit='all', profile_flow_control='scripts_only',
         profile_mixed_precision=c['dtype'],
         profile_fixed_batch_size=1 if kind=='computation' else 16,
         profile_fixed_seq_length_list=[c['seq']], profile_layernum_min=1,
         profile_layernum_max=2, profile_max_tp_deg=c['max_tp'],
         profile_dp_type='ddp', runtime_yaml_template_path=str(template))
    for k,v in r['model'].items(): setattr(pa.model_info,k,v)
    pa.model_info.model_config_path = str(model_template)
    os.environ.update(NUM_NODES='2', NUM_GPUS_PER_NODE='8',
                      RUNTIME_LAUNCHER='GALV6_NATIVE_ENTRY')
    (gpt/'scripts').mkdir(parents=True, exist_ok=True)
    (gpt/'configs').mkdir(exist_ok=True)
    profiler = ModelProfiler(pa)
    profiler.set_profiler_launcher(str(gpt), r['model']['model_size'])
    profiler.launch_profiling_scripts()
    script = gpt/'scripts'/f'{kind}_profile_scripts_all.sh'
    commands = []
    for line in script.read_text().splitlines():
        if 'GALV6_NATIVE_ENTRY' not in line: continue
        tokens = shlex.split(line.split('2>&1 | tee',1)[0])
        pos = tokens.index('GALV6_NATIVE_ENTRY')
        env = dict(t.split('=',1) for t in tokens[:pos])
        argv = tokens[pos+1:]
        values = dict(t.split('=',1) for t in argv[1:])
        pp = int(values.get('runtime.parallel.pp_deg',1))
        tp = int(values.get('runtime.parallel.global_tp_deg',1))
        world = 1 if kind=='computation' else 16
        gbs = int(values['runtime.train.global_batch_size'])
        chunks = int(values['runtime.train.chunks'])
        if world % (pp*tp) or gbs % ((world//(pp*tp))*chunks):
            raise RuntimeError(f'Invalid native Profile batch/world configuration: {line}')
        commands.append((argv,env))
    if not commands:
        raise RuntimeError(f'No native {kind} profile commands generated')
    return pa, profiler, commands


def search_timings(out):
    timings = {p.name.removesuffix('.done.json'):json.loads(p.read_text())['seconds']
               for p in (out/'stages').glob('*.done.json')}
    processing = sum(json.loads((out/f'{k}_processing.json').read_text())['seconds']
                     for k in ('computation','memory'))
    seconds = sum(v for k,v in timings.items()
                  if k.startswith(('computation_','memory_')) or k=='search') + processing
    return dict(stage_seconds=timings, profile_processing_s=processing,
                profile_and_search_s=seconds,
                timing_note='Compute/memory Profile launches, processing and search wall time; includes their process startup. Excludes hardware communication calibration, setup, queueing and selected training.')


def make_search_args(c, r, gpt, hardware, out):
    from galvatron.core.search_engine.args_schema import GalvatronSearchArgs
    a = GalvatronSearchArgs()
    for k, v in r['model'].items():
        setattr(a.model_info, k, v)
    a.parallelism_info.mixed_precision = c['dtype']
    a.parallelism_info.default_dp_type = 'ddp'
    a.parallelism_info.pipeline_type = 'pipedream_flush'
    a.common_train_info.seq_length = c['seq']
    a.common_train_info.global_batch_size = c['gbs']
    a.common_train_info.sequence_parallel = True
    a.hardware_info.num_nodes = 2
    a.hardware_info.num_gpus_per_node = 8
    a.hardware_info.memory_constraint = 28
    a.batch_size_info.settle_bsz = c['gbs']
    a.search_space_info.disable_ckpt = 1
    a.search_space_info.disable_fsdp = 1
    a.search_space_info.disable_cp = 1
    # Here "sp" denotes independent sequence/Ulysses partitioning, NOT TP-SP.
    a.search_space_info.disable_sp = 1
    a.search_space_info.disable_embedding_lmhead_sp = 1
    a.search_space_info.max_tp_deg = c['max_tp']
    a.search_space_info.max_pp_deg = 8
    a.options_info.fine_grained_mode = 0
    a.options_info.output_config_path = str(out/'selected')
    a.options_info.log_dir = str(out/'search_logs')
    a.profiling_info.memory_profiling_path = str(gpt/'configs')
    a.profiling_info.time_profiling_path = str(gpt/'configs')
    for field in ('allreduce_bandwidth_config_path','p2p_bandwidth_config_path','overlap_coe_path','sp_time_path'):
        setattr(a.profiling_info, field, str(hardware/'hardware_configs'))
    return a


def common_space_preflight(c, r, gpt, hardware, out):
    """Check strategy support before communication or model profiling."""
    from common5_space import require_equal, signature
    from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
    a = make_search_args(c, r, gpt, hardware, out)
    engine = GalvatronSearchEngine(a)
    engine.total_layernum = c['layers']
    engine.generate_strategy_list()
    engine.filter_strategy_list()
    supported = {(s.dp_size,s.pp_size,s.tp_size,mbs)
                 for s in engine.layer_strategy_list
                 if s.cp_size == 1 and s.sp_size == 1 and not s.checkpoint
                 and getattr(s.dp_type,'name',str(s.dp_type)).upper() == 'DDP'
                 for mbs in (1,2,4,8)
                 if c['gbs'] % (s.dp_size*mbs) == 0
                 and c['gbs']//(s.dp_size*mbs) >= s.pp_size}
    expected = common_rows(c)
    require_equal(supported, expected, 'Galvatron expressible strategies')
    write(out/'common_space.json', dict(count=len(expected),sha256=signature(expected),
                                       candidates=expected,memory_limit_gib=28,
                                       status='preflight_passed'))


def search(c, r, gpt, hardware, out):
    from common5_space import signature
    from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
    from galvatron.utils.hf_config_adapter import model_layer_configs, model_name
    a = make_search_args(c, r, gpt, hardware, out)
    (out/'selected').mkdir(exist_ok=True)
    write(out/'search_args.json', a.model_dump())
    engine = GalvatronSearchEngine(a)
    engine.set_search_engine_info(path=str(gpt), model_layer_configs=model_layer_configs(a), model_name=model_name(a))
    engine.initialize_search_engine(show_all_strategy_list=False)
    all_layer = engine.layer_strategy_list
    all_embedding = engine.embedding_lmhead_strategy_list
    rows = common_rows(c)
    winner = None
    records = []
    for dp,pp,tp,mbs in rows:
        print('COMMON CANDIDATE',len(records)+1,'/',len(rows),(dp,pp,tp,mbs),flush=True)
        chunks = c['gbs']//(dp*mbs)
        engine.layer_strategy_list = [s for s in all_layer if
            (s.dp_size,s.pp_size,s.tp_size,s.sp_size,s.cp_size)==(dp,pp,tp,1,1)
            and not s.checkpoint and getattr(s.dp_type,'name',str(s.dp_type)).upper()=='DDP']
        engine.embedding_lmhead_strategy_list = [s for s in all_embedding if
            (s.dp_size,s.pp_size,s.tp_size,s.sp_size,s.cp_size)==(dp,pp,tp,1,1)
            and getattr(s.dp_type,'name',str(s.dp_type)).upper()=='DDP']
        if len(engine.layer_strategy_list)!=1 or len(engine.embedding_lmhead_strategy_list)!=1:
            raise RuntimeError(f'Expected one Galvatron strategy for {(dp,pp,tp,mbs)}; '
                f'found {len(engine.layer_strategy_list)} layer, '
                f'{len(engine.embedding_lmhead_strategy_list)} embedding')
        try:
            result = engine.search_for_single_task(c['gbs'],chunks,pp,tp,'tp_with_sp')
        except Exception as exc:
            records.append(dict(candidate=[dp,pp,tp,mbs],error=repr(exc)))
            write(out/'candidate_evaluations.json',dict(common_candidate_count=len(rows),
                  common_candidate_sha256=signature(rows),records=records,
                  status='interrupted'))
            raise
        cost = float(result.get('time_cost',float('inf')))
        record = dict(candidate=[dp,pp,tp,mbs],chunks=chunks,
                      throughput=float(result['throughput']),
                      time_cost=cost if math.isfinite(cost) else None,
                      memory_cost=result.get('memory_cost'))
        records.append(record)
        write(out/'candidate_evaluations.json',dict(common_candidate_count=len(rows),
              common_candidate_sha256=signature(rows),records=records,
              status='running'))
        if math.isfinite(record['throughput']) and record['throughput']>0 and (
            winner is None or record['throughput']>winner[0]):
            winner=(record['throughput'],result,chunks,(dp,pp,tp,mbs))
    write(out/'candidate_evaluations.json',dict(common_candidate_count=len(rows),
          common_candidate_sha256=signature(rows),records=records,status='completed'))
    if winner:
        engine.save_results(winner[1],c['gbs'],winner[2])
        write(out/'selected_candidate.json',dict(candidate=winner[3],chunks=winner[2]))
    else:
        write(out/'selected_candidate.json',dict(candidate=None,
              reason='Galvatron predicted no feasible candidate'))


def run_workflow(index):
    import fcntl
    attempt_start = time.perf_counter()
    repo = Path(os.environ['GALV_REPO']).resolve()
    c = json.loads((HERE/'cases.json').read_text())[index]
    out = repo/'mm_logs'/os.environ['GALV6_TAG']/c['id']
    out.mkdir(parents=True, exist_ok=True)
    resumed = any((out/'stages').glob('*.done.json'))
    lock = (out/'RUNNING.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    r = runtime_config(c, Path(os.environ['MEGATRON_ROOT']))
    contract = dict(case=c, runtime=r, revision=REVISION,
                    nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']], text=True).split(),
                    kit_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()})
    cp = out/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise RuntimeError('Run contract changed (including node allocation); choose a new GALV6_TAG')
    write(cp, contract)
    if (out/'summary.json').exists():
        print('ALREADY COMPLETE', out, flush=True)
        return
    preflight(c, r, repo, out)
    common_space_preflight(c, r, out/'gpt', out/'hardware', out)
    # Isolate native profiler outputs; no concurrent writes into the upstream repo.
    gpt, hw = out/'gpt', out/'hardware'
    gpt.mkdir(exist_ok=True)
    for folder in ('scripts','configs'):
        (gpt/folder).mkdir(exist_ok=True)
    for name in ('train_dist.py','search_dist.py'):
        target = gpt/name
        if not target.exists():
            shutil.copy2(repo/'galvatron/models/gpt'/name, target)
    hw.mkdir(exist_ok=True)
    (hw/'hardware_configs').mkdir(exist_ok=True)
    for p in (repo/'galvatron/profile_hardware').glob('*.py'):
        if not (hw/p.name).exists(): shutil.copy2(p, hw/p.name)
    runtime = out/'runtime.yaml'
    yaml_write(runtime, {'runtime':r})
    baseenv = dict(GALV6_CASE=c['id'], GALV6_QK='1' if c['qk_norm'] else '0')
    # Communication calibration uses upstream tests and their real outputs.
    batches = ['1024','512','256','128','64','32','16','8','4','2','1']
    hw_jobs = [
      ('allreduce', 'profile_allreduce.py', ['--global_tp_deg','16','8','4','2','--profile_time','0'], 2,8),
      ('p2p', 'profile_p2p.py', ['--pp_deg','2','4','8'], 2,8),
      ('sp_allreduce','profile_allreduce.py',['--global_tp_deg','8','4','2','--local_batch_size']+batches+['--profile_time','1'],2,8),
      ('sp_all2all','profile_all2all.py',['--global_tp_deg','8','4','2','--local_batch_size']+batches,2,8),
      ('overlap','profile_overlap.py',['--overlap_time_multiply','4'],1,8)]
    for name, script, argv, nn, ng in hw_jobs:
        stage(out, 'hardware_'+name, [str(hw/script)]+argv, hw, nn, ng)
    # Generate commands from the pinned native profiler, execute with Slurm.
    for kind in ('computation','memory'):
        pa, profiler, commands = prepare_profiles(c,r,out,gpt,kind)
        print(f'{kind}: {len(commands)} native profiling launches', flush=True)
        for i,(argv,native_env) in enumerate(commands):
            # Native YAML overrides stay unchanged; source output path is isolated gpt.
            stage(out, f'{kind}_{i:03d}', [str(HERE/'run_six.py'),'train',
                  '--source',str(gpt/'train_dist.py'),'--config',argv[0], '--']+argv[1:],
                  gpt, 1 if kind=='computation' else 2,
                  1 if kind=='computation' else 8, dict(baseenv,**native_env))
        pa.profile_flow_control = 'data_only'
        processing_started = time.perf_counter()
        profiler.process_profiled_data()
        write(out/f'{kind}_processing.json',
              dict(seconds=time.perf_counter()-processing_started))
    # Search in a separate process so its logs and elapsed time are isolated.
    stage(out, 'search', [sys.executable,'-u',str(HERE/'run_six.py'),
                         'search','--out',str(out),'--index',str(index)], gpt)
    selected = list((out/'selected').glob('galvatron_config_*.json'))
    if not selected and json.loads((out/'selected_candidate.json').read_text())['candidate'] is None:
        write(out/'summary.json',dict(case=c['id'],status='no_feasible_candidate',
            common_candidate_count=len(common_rows(c)),memory_limit_gib=28,
            **search_timings(out)))
        write(out/'status.json',dict(status='no_feasible_candidate'))
        return
    if len(selected) != 1:
        raise RuntimeError(f'Expected one native search winner; found {len(selected)}')
    winner = json.loads(selected[0].read_text())
    assert winner['global_bsz'] == c['gbs'], winner
    chosen = json.loads((out/'selected_candidate.json').read_text())['candidate']
    assert chosen is not None and chosen in [list(row) for row in common_rows(c)]
    assert winner['pp_deg']==chosen[1] and winner['chunks']==c['gbs']//(chosen[0]*chosen[3])
    assert winner['world_size']==16 and winner['vtp']==chosen[2] and winner['vsp']==0
    assert set(map(int,winner['tp_sizes_enc'].split(',')))=={chosen[2]}
    assert set(map(int,winner['dp_types_enc'].split(',')))=={0}
    assert set(map(int,winner['use_sp'].split(',')))=={0}
    assert set(map(int,winner['checkpoint'].split(',')))=={0}
    assert len(winner['tp_sizes_enc'].split(',')) == c['layers']
    assert list(map(int,winner['pp_division'].split(','))) == [c['layers']//chosen[1]]*chosen[1]
    assert winner['embed_sdp']==0 and winner['default_dp_type']=='ddp'
    r['parallel']['galvatron_config_path'] = str(selected[0])
    r['train']['chunks'] = winner['chunks']
    r['train']['micro_batch_size'] = chosen[3]
    r['train']['sequence_parallel'] = chosen[2] > 1
    yaml_write(runtime, {'runtime':r})
    stage(out, 'selected_training', [str(HERE/'run_six.py'),'train',
          '--source',str(gpt/'train_dist.py'),'--config',str(runtime),'--out',str(out)],
          gpt, 2, 8, baseenv)
    training = json.loads((out/'training_summary.json').read_text())
    timing = search_timings(out)
    write(out/'summary.json', dict(case=c['id'], status='completed', selected=winner,
         training=training, **timing,
         native_stage_total_s=sum(timing['stage_seconds'].values()),
         uninterrupted_workflow_wall_s=None if resumed else time.perf_counter()-attempt_start,
         resumed=resumed,
         comparison_scope='Common5 explicit DP/TP/PP/MBS candidates; native Galvatron cost model; compare evaluated sets using collect_exact.py'))
    write(out/'status.json', dict(status='completed'))
    print('COMPLETE',out/'summary.json',flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('run'); p.add_argument('--index',type=int,choices=(0,1,2,3,5),required=True)
    p=sub.add_parser('worker'); p.add_argument('spec')
    p=sub.add_parser('search'); p.add_argument('--out',required=True); p.add_argument('--index',type=int,required=True)
    p=sub.add_parser('space-check'); p.add_argument('--index',type=int,choices=(0,1,2,3,5))
    p=sub.add_parser('train'); p.add_argument('--source',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out'); p.add_argument('overrides',nargs=argparse.REMAINDER)
    a=parser.parse_args()
    if a.mode=='worker': worker(a.spec)
    elif a.mode=='space-check':
        import tempfile
        cases=json.loads((HERE/'cases.json').read_text())
        for index in ((a.index,) if a.index is not None else (0,1,2,3,5)):
            c=cases[index]
            r=runtime_config(c,Path(os.environ['MEGATRON_ROOT']))
            with tempfile.TemporaryDirectory(prefix='common5-space-') as temp:
                out=Path(temp)
                common_space_preflight(c,r,out/'gpt',out/'hardware',out)
                for kind in ('computation','memory'):
                    _,_,commands=prepare_profiles(c,r,out,out/'gpt',kind)
                    print('PROFILE COMMAND PASS',c['id'],kind,len(commands),flush=True)
            print('SPACE PASS',c['id'],len(common_rows(c)),flush=True)
    elif a.mode=='train':
        overrides=a.overrides[1:] if a.overrides[:1]==['--'] else a.overrides
        train_entry(a.config,overrides,a.source,a.out,os.environ.get('GALV6_QK')=='1')
    elif a.mode=='search':
        out=Path(a.out); c=json.loads((HERE/'cases.json').read_text())[a.index]
        r=json.loads((out/'runtime.yaml').read_text())['runtime']
        search(c,r,out/'gpt',out/'hardware',out)
    else: run_workflow(a.index)


if __name__=='__main__':
    main()
