#!/usr/bin/env python3
"""CPU-only declaration generator, NOT a search or training adapter.

Generate the same finite eight-GPU DP/TP/PP/MBS input list for both systems.
Every row remains memory-unknown and backend-unverified. Runtime adapters must
independently record and compare their effective models and consumed rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 8
MODEL_KEYS = (
    'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
    'native_kv_heads', 'seq_length', 'global_batch_size', 'dtype',
    'declared_vocab_size', 'rotary_base', 'norm_epsilon',
    'untie_embeddings_and_output_weights', 'add_qkv_bias', 'qk_layernorm',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def candidate_signature(rows):
    """Same row-order-independent encoding as the old common5_space helper."""
    return hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def candidates(case, world=WORLD_SIZE):
    if world != WORLD_SIZE:
        raise ValueError('This H20 contract is fixed to eight GPUs')
    for key in ('num_layers', 'hidden_size', 'ffn_hidden_size',
                'num_attention_heads', 'native_kv_heads', 'seq_length',
                'global_batch_size'):
        if type(case[key]) is not int or case[key] <= 0:
            raise ValueError('Invalid positive integer: ' + key)
    if case['num_attention_heads'] % case['native_kv_heads']:
        raise ValueError('Query heads must be divisible by native KV heads')
    if case['hidden_size'] % case['num_attention_heads']:
        raise ValueError('Hidden size must be divisible by query heads')
    rows = []
    for pp in (1, 2, 4, 8):
        if case['num_layers'] % pp:
            continue
        for tp in (1, 2, 4, 8):
            if (pp * tp > world or world % (pp * tp)
                    or case['hidden_size'] % tp
                    or case['num_attention_heads'] % tp
                    or case['native_kv_heads'] % tp):
                continue
            dp = world // (pp * tp)
            for mbs in (8, 4, 2, 1):
                gbs = case['global_batch_size']
                if gbs % (dp * mbs) or gbs // (dp * mbs) < pp:
                    continue
                rows.append((dp, pp, tp, mbs))
    if len(rows) != len(set(rows)):
        raise ValueError('Duplicate structural candidates')
    return rows


def require_equal(observed, expected, label):
    """Adapter utility: set equality alone must not conceal duplicates."""
    observed, expected = list(map(tuple, observed)), list(map(tuple, expected))
    if len(observed) != len(set(observed)) or len(expected) != len(set(expected)):
        raise ValueError(label + ': duplicate candidate rows')
    if set(observed) != set(expected):
        raise ValueError(f'{label}: missing={sorted(set(expected)-set(observed))}; '
                         f'extra={sorted(set(observed)-set(expected))}')


def declared_model_spec(config, case):
    return {'fixed_model_defaults': config['fixed_model_defaults'],
            'model': {key: case[key] for key in MODEL_KEYS},
            'declared_inputs': case['inputs'],
            'unresolved_audit_notes': case.get('audit_notes', []),
            'model_contract_status': case['model_contract_status']}


def validate(config):
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported case schema')
    target = config['hardware_target']
    if (target['nodes'], target['gpus_per_node'], target['world_size']) != (1, 8, 8):
        raise ValueError('Expected one node with eight GPUs')
    space = config['common_space']
    for key in ('tp', 'pp', 'mbs'):
        if space[key] != [1, 2, 4, 8]:
            raise ValueError('Unreviewed search dimension: ' + key)
    if space['sp_degree'] != 'tp' or space['runtime_sequence_parallel'] != 'tp > 1':
        raise ValueError('SP policy must follow TP')
    if any(space[k] != 1 for k in ('cp', 'up', 'ep')):
        raise ValueError('Common space fixes CP/UP/EP to one')
    if not space['uniform_pipeline_layers'] or any(space[k] for k in (
            'recompute', 'distributed_optimizer', 'vpp', 'structural_gqa_search')):
        raise ValueError('Common space must preserve the no-extra-optimization policy')
    m = config['measurement_protocol']
    if (m['independent_runs'], m['total_iterations'], m['discard_first'],
            m['measured_iterations']) != (1, 10, 5, [6, 7, 8, 9, 10]):
        raise ValueError('Expected one run, ten steps, measure steps six through ten')
    if (m['hardware_communication_profile_in_search_time']
            or not m['compute_and_memory_profile_in_search_time']
            or m['training_time_in_search_time']):
        raise ValueError('Search timing boundary has changed')
    cases = config['cases']
    if len(cases) != 8 or len({c['id'] for c in cases}) != 8:
        raise ValueError('Expected eight distinct workloads')
    total = 0
    for case in cases:
        if not re.fullmatch(r'[a-z0-9_]+', case['id']):
            raise ValueError('Unsafe case id')
        if case['dtype'] not in ('fp16', 'bf16'):
            raise ValueError('Unexpected precision')
        rows = candidates(case)
        if case['hidden_size'] // case['num_attention_heads'] != config['fixed_model_defaults']['head_dim']:
            raise ValueError('Unexpected attention head dimension')
        declared_model_spec(config, case)
        for key in ('data_prefix', 'megatron_tokenizer', 'galvatron_tokenizer'):
            name = case['inputs'][key]
            if name.startswith(('/', '\\')) or ':' in name or '..' in Path(name).parts:
                raise ValueError('Input path must be repository-relative: ' + name)
        count = len(rows)
        if count != case['expected_structural_candidates']:
            raise ValueError(case['id'] + ': structural count changed')
        total += count
    if total != space['expected_total_structural_candidates'] or total != 296:
        raise ValueError('Expected 296 structurally legal rows across eight workloads')


def build_payload(config, case):
    rows = candidates(case)
    declared = declared_model_spec(config, case)
    records = []
    for index, (dp, pp, tp, mbs) in enumerate(rows):
        chunks = case['global_batch_size'] // (dp * mbs)
        records.append({
            'candidate_id': f"{case['id']}_{index:03d}",
            'row': [dp, pp, tp, mbs], 'dp': dp, 'pp': pp, 'tp': tp, 'mbs': mbs,
            'sp_degree': tp, 'runtime_sequence_parallel': tp > 1,
            'microbatch_count': chunks,
            'megatron_parallel_encoding': [dp, pp, 1, 1, tp, tp, 1, mbs, chunks],
            'strategy': {'ReCompute': [None, None], 'VirtualPipe': None,
                         'DistributedOptimizer': False,
                         'Hybrid_MHA_MQA': [case['native_kv_heads'] != case['num_attention_heads'],
                                            case['native_kv_heads']]},
            'structurally_legal': True, 'modeled_memory_feasible': None,
            'backend_support_verified': False, 'training_status': 'not_run',
        })
    return {'schema_version': 1, 'case_id': case['id'], 'world_size': WORLD_SIZE,
            'status': 'structural_manifest_only',
            'row_fields': ['dp', 'pp', 'tp', 'mbs'],
            'structural_candidate_count': len(rows),
            'candidate_sha256': candidate_signature(rows),
            'declared_model_spec': declared,
            'declared_model_spec_sha256': digest(declared),
            'hash_scope': 'Declaration only: excludes actual input bytes, parsed backend arguments and runtime equality validation.',
            'measurement_protocol': config['measurement_protocol'],
            'memory_limit_gib': config['hardware_target']['memory_limit_gib'],
            'actual_model_equality_verified': False,
            'actual_candidate_consumption_verified': False,
            'candidates': records}


def generate(config_path, output):
    config_path, output = Path(config_path), Path(output)
    raw = config_path.read_bytes()
    config = json.loads(raw.decode('utf-8'))
    validate(config)
    payloads = [build_payload(config, case) for case in config['cases']]
    # Refuse even an empty existing directory. No old results are overwritten.
    output.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': 1, 'status': 'structural_manifest_only',
               'config_file_sha256': hashlib.sha256(raw).hexdigest(),
               'hardware_target': config['hardware_target'],
               'total_structural_candidates': sum(p['structural_candidate_count'] for p in payloads),
               'modeled_memory_feasible_candidates': None,
               'backend_adapters_validated': False, 'gpu_training_performed': False,
               'cases': []}
    for payload in payloads:
        name = payload['case_id'] + '.json'
        with (output / name).open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        summary['cases'].append({key: payload[key] for key in (
            'case_id', 'structural_candidate_count', 'candidate_sha256',
            'declared_model_spec_sha256')})
    with (output / 'manifest.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'config' / 'cases8.json')
    parser.add_argument('--out', type=Path, required=True,
                        help='A new directory; existing paths are rejected')
    args = parser.parse_args()
    try:
        summary = generate(args.cases, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(f"STRUCTURAL MANIFEST: {args.out.resolve()}")
    print(f"{summary['total_structural_candidates']} rows across eight workloads; memory feasibility and GPU execution are NOT validated.")


if __name__ == '__main__':
    main()
