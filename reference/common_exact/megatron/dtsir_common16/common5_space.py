"""The shared, finite 16-GPU DP/TP/PP candidate contract."""

import hashlib
import json


def candidates(*, layers, hidden, heads, kv_heads, global_batch, world=16):
    if world != 16:
        raise ValueError('Common5 is a fixed 16-GPU protocol')
    rows = []
    for pp in (1, 2, 4, 8):
        if layers % pp:
            continue
        for tp in (1, 2, 4, 8):
            if pp * tp > world or hidden % tp or heads % tp or kv_heads % tp:
                continue
            dp = world // (pp * tp)
            for mbs in (8, 4, 2, 1):
                if global_batch % (dp * mbs) or global_batch // (dp * mbs) < pp:
                    continue
                rows.append((dp, pp, tp, mbs))
    assert rows and len(rows) == len(set(rows))
    return rows


def megatron_parallel(row, global_batch):
    dp, pp, tp, mbs = row
    return [dp, pp, 1, 1, tp, tp, 1, mbs, global_batch // (dp * mbs)]


def signature(rows):
    return hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest()


def require_equal(observed, expected, label):
    observed, expected = list(map(tuple, observed)), list(map(tuple, expected))
    if len(observed) != len(set(observed)) or len(expected) != len(set(expected)):
        raise RuntimeError(f'{label}: duplicate candidate records')
    observed, expected = set(observed), set(expected)
    if observed != expected:
        raise RuntimeError(
            f'{label}: missing={sorted(expected - observed)} extra={sorted(observed - expected)}'
        )
