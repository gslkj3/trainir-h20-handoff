"""Use upstream per-layer RoPE for selected Ulysses layouts on real data."""
import json
from pathlib import Path


def needs_native_layer_rope(args):
    path=args.parallel.galvatron_config_path
    if not path or args.model.position_embedding_type!='rope':return False
    config=json.loads(Path(path).read_text())
    return any(int(x) for x in config.get('use_sp','0').split(','))


def use_native_layer_rope(batch):
    tokens,kwargs,loss_func=batch
    kwargs=dict(kwargs)
    # GalvatronAttention.forward(None) invokes upstream _get_rotary_pos_emb,
    # which uses each layer's actual TP/SP groups and sequence offset.
    kwargs['rotary_embedding']=None
    return tokens,kwargs,loss_func
