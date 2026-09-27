"""Opt-in diagnostics. Observations are not timing benchmarks or full traces."""
import atexit
import collections
import json
import os
from pathlib import Path
import time
import hashlib


def enable_fixed_prediction_stats(rt):
    # The launcher's measure mode must remain unchanged for real training.
    # Only the rank-0 fixed evaluator gets an active statistics configuration.
    rt.config.experiment = 'ir_fixed_prediction'
    rt.config.profile_seed_from_existing = True
    rt.config.capture_candidates = True
    rt.config.capture_rejected = True
    rt.set_variant('no_inc')
    rt.reset()
    if not rt.config.active or rt.config.use_incremental or not rt.config.use_profile_reuse:
        raise RuntimeError('Fixed prediction requires active stats, no incremental reuse, and Profile reuse.')


def evidence_files(root):
    result = {}
    for relative in ('calc_data/data.json', 'comm_data/profile_comm.json'):
        path = Path(root) / relative
        result[relative] = {'path': str(path.resolve()), 'exists': path.is_file()}
        if path.is_file():
            result[relative]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def save(name, data):
    out = Path(os.environ['DTSIR_IR_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=2, default=str), encoding='utf8')


def predict_fixed(args):
    import torch.distributed as dist
    from test_parallel_model import GPT, TPDS_RUNTIME as rt
    if dist.get_rank() == 0:
        record = json.loads(os.environ['DTSIR_MEASURE_CANDIDATE_JSON'])
        s = record['parallel']
        rt.refresh(os.getenv('DTSIR_MML_LOGS', 'mm_logs'))
        enable_fixed_prediction_stats(rt)
        rt.note_strategy(args)
        evidence_root = os.getenv('DTSIR_MML_LOGS', 'mm_logs')
        input_evidence = evidence_files(evidence_root)
        started = time.perf_counter()
        model = GPT(args, mmlogs_path=os.getenv('DTSIR_MML_LOGS', 'mm_logs'), search_level=4)
        # Use the existing model, but screen exactly this configuration.
        model._tpds_enumerate_structural_candidates = lambda: [s]
        feasible = model.search_space_create()
        peak = max(float(model.eval_calc(x, s)) for x in model.memory_model)
        cost = model.costmodel_create(feasible)[0] if feasible else None
        rt.timings['fixed_prediction_total_seconds'] = time.perf_counter()-started
        # Keep newly measured entries available to later fixed candidates, as in
        # the original non-instrumented cache path. FileJSONHandler saves at exit.
        for (name, key), value in rt.profile_overlay.items():
            model.map_manager.data_map.setdefault(name, {})[key] = value
        save('prediction.json', {
            'collector_version': '2026-09-20-fixed-stats-p2psafe',
            'input_evidence': input_evidence,
            'candidate': record, 'feasible': bool(feasible),
            'predicted_peak_bytes': peak, 'predicted_cost_native_units': cost,
            'memory_components_native': model.memory_model_analysis,
            'execution_graph': model.cost_model,
            'model_dimensions': {k: getattr(model, k, None) for k in ('layer','seq','hidden','head','group','h_ffn')},
            'diagnostic_wall_seconds': time.perf_counter()-started,
            'runtime': rt.result_dict(),
            'profile_accounting': {
                'queries': rt.stats['profile_queries'],
                'hits': rt.stats['profile_hits'],
                'disk_seed_hits': rt.stats['profile_seed_hits'],
                'new_measurements': rt.stats['profile_measurements'],
                'unique_used_keys': len(rt.profile_overlay),
                'unique_newly_measured_keys': len(rt.profile_unique_measured),
                'note': 'Warm existing operator evidence is allowed. Zero new measurements can be valid; queries should be nonzero for this feasible model.'},
            'scope': 'Existing model prediction, not independent ground truth. User reports deliberate environment compensation for DP synchronization; preserve it and verify compensated predictions against observations.',
            'environment_compensation': os.getenv('DTSIR_COMPENSATION_NOTE', 'User-reported compensation; details pending')})
    dist.barrier()


COLLECTIVE_OBSERVER_NAMES = (
    'all_reduce', 'all_gather', 'all_gather_into_tensor',
    'reduce_scatter', 'reduce_scatter_tensor', 'all_to_all_single',
)


def install_collective_observers(dist, tensors, comm, max_rows):
    # P2POp validates function identity. Never replace isend/irecv or P2P APIs.
    for name in COLLECTIVE_OBSERVER_NAMES:
        original = getattr(dist, name, None)
        if original is None:
            continue
        def wrapped(*a, _name=name, _original=original, **kw):
            signature = json.dumps({'operation': _name, 'tensor_arguments': tensors(a),
                                    'tensor_keywords': tensors(kw)}, sort_keys=True)
            if signature in comm or len(comm) < max_rows:
                comm[signature] += 1
            return _original(*a, **kw)
        setattr(dist, name, wrapped)


def install_observer(args):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank()
    modules = {}
    comm = collections.Counter()
    seen_parameters = set()
    parameter_bytes = 0
    max_rows = 6000

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            return [{'shape': list(obj.shape), 'dtype': str(obj.dtype), 'bytes': obj.numel()*obj.element_size()}]
        if isinstance(obj, (list, tuple)):
            return [t for x in obj for t in tensors(x)]
        if isinstance(obj, dict):
            return [t for x in obj.values() for t in tensors(x)]
        return []

    def hook(module, inputs, output):
        nonlocal parameter_bytes
        typ = type(module).__module__ + '.' + type(module).__name__
        if not any(x in typ.lower() for x in ('attention','linear','transformer','mlp','norm')):
            return
        key = str(id(module))
        if key not in modules and len(modules) < max_rows:
            params = []
            for name, p in module.named_parameters(recurse=False):
                params.append({'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype)})
                if id(p) not in seen_parameters:
                    parameter_bytes += p.numel()*p.element_size()
                    seen_parameters.add(id(p))
            modules[key] = {'type': typ, 'inputs': tensors(inputs), 'outputs': tensors(output),
                            'parameters': params, 'calls_grad': 0, 'calls_no_grad': 0}
        if key in modules:
            field = 'calls_grad' if torch.is_grad_enabled() else 'calls_no_grad'
            modules[key][field] += 1

    torch.nn.modules.module.register_module_forward_hook(hook)
    # Python entry-point coverage only: fused/C++/previously imported aliases can bypass this.
    install_collective_observers(dist, tensors, comm, max_rows)

    save(f'rank{rank}_started.json', {'rank': rank, 'effective_args': vars(args),
         'torch': torch.__version__, 'cuda': torch.version.cuda,
         'device': torch.cuda.get_device_name(), 'candidate': json.loads(os.environ['DTSIR_MEASURE_CANDIDATE_JSON'])})
    def finish():
        save(f'rank{rank}_observations.json', {
            'rank': rank, 'modules': list(modules.values()),
            'observer_version': '2026-09-20-p2pfix',
            'communication_coverage': {
                'wrapped_python_collectives': list(COLLECTIVE_OBSERVER_NAMES),
                'p2p_observed': False,
                'note': 'P2P functions are deliberately unmodified. Missing P2P observations do not mean zero P2P communication.'},
            'communication_python_calls': [{'signature': json.loads(k), 'count': v} for k,v in comm.items()],
            'observed_unique_parameter_bytes': parameter_bytes,
            'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'process_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scope': 'Process-lifetime peaks; partial Python communication/module coverage. Call counts are NOT certified replay counts; parameter bytes exclude unseen modules and optimizer states. Instrumented timings must not be used as clean training throughput.'})
    atexit.register(finish)
