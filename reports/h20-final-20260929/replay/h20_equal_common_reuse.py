"""User-authorized timing proxy for identical common winners; never fake a Megatron run."""
import json,hashlib
from pathlib import Path
FIELDS=['dp','pp','cp','up','tp','sp','ep','mbs','chunks']
def decide(root,case,space,selected):
 if space!='common':return dict(eligible=False,reason='full_space_requires_measurement')
 g=next(x for x in json.loads((root/'galvatron_reuse.json').read_text()) if x['case']==case['id'] and x['space']=='common')
 signature=dict(zip(FIELDS,selected['parallel']));signature['kv_heads']=selected['strategy']['Hybrid_MHA_MQA'][1]
 reference={k:g.get(k) for k in signature};st=selected['strategy'];gs=json.loads(g['selected_strategy']);reasons=[]
 if signature!=reference:reasons.append('parallel_batch_or_kv_mismatch')
 if st['ReCompute']!=[None,None] or st['VirtualPipe'] is not None or st['DistributedOptimizer']:reasons.append('noncommon_megatron_strategy')
 if any(int(x) for x in gs['checkpoint'].split(',')) or any(int(x) for x in gs['dp_types_enc'].split(',')) or gs.get('embed_sdp')!=0 or gs.get('default_dp_type')!='ddp':reasons.append('noncommon_galvatron_strategy')
 if set(map(int,gs['tp_sizes_enc'].split(',')))!={signature['tp']}:reasons.append('galvatron_layer_tp_differs')
 if list(map(int,gs['pp_division'].split(',')))!=[case['num_layers']//signature['pp']]*signature['pp']:reasons.append('different_layer_partition')
 if g['status']!='completed' or not g.get('reuse_verified'):reasons.append('unverified_reference')
 evidence=Path(g['training_evidence']);summary=evidence/'training_summary.json'
 return dict(eligible=not reasons,reasons=reasons,signature=signature,reference_signature=reference,
   case=case['id'],space=space,reference_system='galvatron',reference_campaign=g['reused_from_campaign'],reference_evidence=str(evidence),reference_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
   policy='User-authorized common-space equal-configuration timing reuse; not an independently measured Megatron throughput or proof of equal backend speed.',
   reused_metrics={k:g[k] for k in ['mean_iteration_s','samples_per_second','tokens_per_second']})
