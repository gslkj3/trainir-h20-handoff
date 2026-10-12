"""Editable scientific plotting source; no smoothing and no hindsight selection."""
import json,csv,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ablation_core import ROOT
p=json.loads((ROOT/'frozen_protocol.json').read_text());snap=json.loads((ROOT/'timepoint_summary.json').read_text());training={r['config_hash']:r for r in json.loads((ROOT/'training_results.json').read_text())};events=json.loads((ROOT/'coverage_events.json').read_text());rows=[]
for r in snap:
 t=training.get(r['best_hash']);row=dict(r,training_status=t['status'] if t else 'no_feasible',mean_step_seconds=t.get('mean_step_seconds') if t else None,samples_per_second=t.get('samples_per_second') if t else None,training_evidence=t.get('evidence') if t else None);rows.append(row)
(ROOT/'final_timepoints.json').write_text(json.dumps(rows,indent=2)+'\n')
with (ROOT/'final_timepoints.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(12,5.0),layout='constrained');labels={'full':'Full','no_inc':'No-Inc','no_profile':'No-Profile','naive':'Naive'};colors=['#0072B2','#E69F00','#009E73','#CC79A7'];markers=['o','s','^','D'];styles=['-','--','-.',':']
for variant,color,marker,style in zip(p['variant_order'],colors,markers,styles):
 e=sorted([r for r in events if r['variant']==variant],key=lambda r:r['time_s']);xs=[0]+[r['time_s'] for r in e]+[p['max_budget_seconds']];ys=[0]+list(range(1,len(e)+1))+[len(e)]
 axes[0].step(xs,ys,where='post',label=labels[variant],color=color,linestyle=style,linewidth=1.6)
 values=[r for r in rows if r['variant']==variant];y=[r['mean_step_seconds'] if r['training_status']=='completed' else float('nan') for r in values]
 axes[1].plot([r['budget_seconds'] for r in values],y,label=labels[variant],color=color,marker=marker,linestyle=style,markersize=8 if variant in ['full','no_profile'] else 5,markerfacecolor='none')
 for r in values:
  if r['training_status']!='completed':
   level=-.13-.07*(p['variant_order'].index(variant)-2)
   axes[1].plot(r['budget_seconds'],level,marker='x',color=color,transform=axes[1].get_xaxis_transform(),clip_on=False)
   missing=[v for v in values if v['training_status']==r['training_status']]
   if r is missing[-1]:
    text=('No feasible: ' if r['training_status']=='no_feasible' else r['training_status']+': ')+labels[variant]
    axes[1].text(r['budget_seconds']+12,level,text,transform=axes[1].get_xaxis_transform(),va='center',fontsize=8,color=color)
for ax in axes:
 ax.set_xlabel('Cumulative search wall time (s)');ax.set_xlim(0,p['max_budget_seconds']);ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel('Distinct completed candidates');axes[0].set_title('(a) Search coverage')
axes[1].set_ylabel('Measured training step time (s; lower is better)');axes[1].set_title('(b) Predictor-selected configuration');axes[1].set_xticks(p['checkpoints_seconds']);axes[1].tick_params(axis='x',labelsize=8)
axes[1].xaxis.labelpad=75
fig.suptitle('LLaMA-7B | sequence 2048 | global batch 256 | 8 × H20',fontsize=12)
for ext in ['pdf','svg','png']:fig.savefig(ROOT/('reuse_ablation.'+ext),dpi=180)
print('Saved PDF, SVG, PNG and final_timepoints.csv; no smoothing or oracle substitution.')
