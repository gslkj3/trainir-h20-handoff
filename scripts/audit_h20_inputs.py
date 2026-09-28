"""Scan all token IDs and indexed-dataset boundaries against local tokenizers."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert not a.out.exists()
root = a.runtime/'Megatron-LM'
rows = []
for family, tokenizer_dir, declared in [('llama', 'llama2-hf', 32000),
                                       ('llama3', 'llama3-hf', 128256),
                                       ('qwen3', 'qwen3-hf', 151936)]:
    tok = AutoTokenizer.from_pretrained(root/'model_from_hf'/tokenizer_dir, local_files_only=True)
    prefix = root/'dataset'/family/'enwiki_text_document'
    index = prefix.with_suffix('.idx')
    binary = prefix.with_suffix('.bin')
    with index.open('rb') as f:
        assert f.read(9) == b'MMIDIDX\x00\x00'
        assert struct.unpack('<Q', f.read(8))[0] == 1
        code = struct.unpack('<B', f.read(1))[0]
        count, documents = struct.unpack('<QQ', f.read(16))
    dtype = {1:'u1', 2:'i1', 3:'<i2', 4:'<i4', 5:'<i8', 8:'<u2'}[code]
    lengths = np.memmap(index, mode='r', dtype='<i4', offset=34, shape=(count,))
    pointers = np.memmap(index, mode='r', dtype='<i8', offset=34+4*count, shape=(count,))
    docs = np.memmap(index, mode='r', dtype='<i8', offset=34+12*count, shape=(documents,))
    ids = np.memmap(binary, mode='r', dtype=dtype)
    assert index.stat().st_size == 34+12*count+8*documents
    assert np.all(lengths >= 0) and pointers[0] == 0
    assert np.array_equal(pointers[1:], pointers[:-1]+lengths[:-1].astype(np.int64)*ids.dtype.itemsize)
    assert int(pointers[-1])+int(lengths[-1])*ids.dtype.itemsize == binary.stat().st_size
    assert int(lengths.sum()) == ids.size
    assert docs[0] == 0 and docs[-1] == count and np.all(docs[1:] >= docs[:-1])
    lower, upper = int(ids.min()), int(ids.max())
    vocab_ids = set(tok.get_vocab().values())
    assert lower >= 0 and upper < len(tok) and upper < declared
    # Validate that even IDs inside the range correspond to actual tokenizer IDs.
    observed = np.zeros(max(vocab_ids)+1, dtype=bool)
    for start in range(0, ids.size, 8_000_000):
        observed[ids[start:start+8_000_000]] = True
    assert set(np.flatnonzero(observed)).issubset(vocab_ids)
    rows.append(dict(family=family, tokenizer=tokenizer_dir, tokenizer_length=len(tok),
                     tokenizer_max_id=max(vocab_ids), declared_embedding_vocab=declared,
                     tokens=int(ids.size), sequences=int(count), document_boundaries=int(documents),
                     dtype=str(ids.dtype), token_min=lower, token_max=upper,
                     tokenizer_eos=tok.eos_token_id, native_padding_divisor1_by_tp={str(tp):((len(tok)+tp-1)//tp)*tp for tp in [1,2,4,8]},
                     passed=True))
    print(family, 'PASS', len(tok), lower, upper, flush=True)
a.out.write_text(json.dumps(dict(passed=True, families=rows,
    note='Qwen historical inputs are preserved. Explicit embedding vocabulary must remain 151936; tokenizer-derived padding alone is not the model contract.'), indent=2))
