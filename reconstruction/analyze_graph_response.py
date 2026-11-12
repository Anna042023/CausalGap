from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reconstruction/analysis'
OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
base=ROOT/'reconstruction/graph_inputs'
h1=pd.read_csv(base/'weak_to_strong.csv');h2=pd.read_csv(base/'calibration.csv')
rows=[]; points=[]
for seed in (7,17,27):
 d=np.load(base/f'graph_input_seed{seed}.npz');y=d['target'];b=d['base_pred']; valid=d['valid'];br=d['base_rep']
 norm=np.linalg.norm(br,axis=-1)
 for label,pct in [('weak05',5),('weak10',10),('strong',20)]:
  pr=d[f'{label}_pred'];rep=d[f'{label}_rep'];q=np.linalg.norm(rep-br,axis=-1)/(norm+1e-7)
  delta=(abs(pr-y)-abs(b-y)).astype(float);delta[~valid]=np.nan
  # per window sensor horizon-average error change
  with np.errstate(invalid='ignore'):
   den=np.sum(valid,axis=1); dsum=np.nansum(delta,axis=1); dcell=np.divide(dsum,den,out=np.full_like(dsum,np.nan),where=den>0)
  # independent sample size is three checkpoints; all granular cells exploratory only
  m=np.isfinite(dcell)&np.isfinite(q)
  qq=q[m];ee=dcell[m]
  rows.append({'checkpoint_seed':seed,'graph_edit_fraction':pct/100,'graph_version':label,'valid_window_sensor_units':len(qq),'mean_relative_rep_response':float(qq.mean()),'median_relative_rep_response':float(np.median(qq)),'mean_mae_change_mph':float(ee.mean()),'mean_absolute_mae_change_mph':float(abs(ee).mean()),'spearman_response_error_change':float(spearmanr(qq,ee).statistic)})
  # aggregated per-sensor over 168 origins only for plotting, one dot per seed x sensor
  for n in range(q.shape[1]):
   if np.isfinite(dcell[:,n]).any():
    points.append({'checkpoint_seed':seed,'graph_edit_percent':pct,'sensor':n,'response':float(np.nanmean(q[:,n])),'mean_mae_change_mph':float(np.nanmean(dcell[:,n]))})
units=pd.DataFrame(rows);pts=pd.DataFrame(points)
units.to_csv(OUT/'response_unit_summary.csv',index=False,float_format='%.12g')
pts.to_csv(OUT/'response_sensor_units.csv',index=False,float_format='%.12g')
# Rebuild rows from R18 per-run source untouched; save frozen copies and aggregation
h1.to_csv(OUT/'weak_to_strong_by_checkpoint.csv',index=False,float_format='%.12g')
h2.to_csv(OUT/'calibration_by_checkpoint.csv',index=False,float_format='%.12g')
s1=h1.groupby('configuration')[['mean_absolute_target_error','mean_squared_target_error']].mean()
s2=h2.groupby('head')['observed_target_mae_mph'].mean()
agg={'H1':{'weak_error_MAE':float(s1.loc['weak_errors','mean_absolute_target_error']),'weak_plus_response_MAE':float(s1.loc['weak_errors_plus_response','mean_absolute_target_error']),'weak_error_MSE':float(s1.loc['weak_errors','mean_squared_target_error']),'weak_plus_response_MSE':float(s1.loc['weak_errors_plus_response','mean_squared_target_error'])},'H2':{k:float(v) for k,v in s2.items()},'graph_response':{'mean_by_edit_pct':{str(k):v for k,v in units.groupby('graph_edit_fraction').mean_mae_change_mph.mean().items()}},'seeds':[7,17,27], 'origins_per_seed':168,'sensors':207,'horizons':12,'h1_test_units_per_seed':33644,'h2_heldout_prediction_cells_per_seed':263296,'status':'source-linked reevaluation; observed fixed targets; not 21-model historical H1/H2'}
(OUT/'response_summary.json').write_text(json.dumps(agg,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
# FIG A: grouped H1 fold MAE/MSE, two panels; small differences emphasized with labels
fig,axes=plt.subplots(1,2,figsize=(7,2.25),constrained_layout=True)
colors={'weak_errors':'#4C78A8','weak_errors_plus_response':'#4C9C89'}
for ax,col,label in zip(axes,['mean_absolute_target_error','mean_squared_target_error'],['Strong-change target MAE (mph)','Strong-change target MSE (mph²)']):
 for config in ['weak_errors','weak_errors_plus_response']:
  dd=h1[h1.configuration==config]
  jitter=-.08 if config=='weak_errors' else .08
  ax.scatter(dd.held_out_checkpoint_seed+jitter,dd[col],s=38,color=colors[config],label=('Weak error only' if config=='weak_errors' else '+ Response') if ax==axes[0] else None,zorder=3)
 for seed,g in h1.groupby('held_out_checkpoint_seed'):
  v=g.set_index('configuration')[col];ax.plot([seed-.08,seed+.08],[v['weak_errors'],v['weak_errors_plus_response']],color='#9BA6AF',lw=1.2,zorder=2)
 ax.set_xticks([7,17,27]);ax.set_xlabel('Held-out checkpoint seed');ax.set_ylabel(label);ax.grid(axis='y',alpha=.20)
axes[0].legend(loc='upper right',frameon=False,fontsize=7)
fig.savefig(FIG/'weak_to_strong_folds.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
# FIG B: point plot, not zero-truncated bars, to avoid implying large relative benefits
fig,ax=plt.subplots(figsize=(6.6,2.4),constrained_layout=True)
heads=['uncorrected','matched_no_response','matched_plus_response','shuffled_response']
colors=['#54768C','#BE866C','#579B84','#999CB3']
x=np.arange(3);w=.16
for j,(head,c) in enumerate(zip(heads,colors)):
 dd=h2[h2['head']==head].sort_values('checkpoint_seed')
 ax.scatter(x+(j-1.5)*w,dd.observed_target_mae_mph,s=47,label=head.replace('_',' ').capitalize(),color=c,marker=['o','^','s','D'][j],zorder=3)
 for xx,yy in zip(x+(j-1.5)*w,dd.observed_target_mae_mph):
  ax.plot([xx,xx],[4.65,yy],color=c,lw=.6,alpha=.25,zorder=1)
ax.set_xticks(x,[7,17,27]);ax.set_xlabel('Checkpoint seed');ax.set_ylabel('Held-out observed-target MAE (mph)')
ax.set_ylim(4.65,5.52);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),frameon=False,ncol=4,fontsize=6.7)
fig.savefig(FIG/'calibration_heads.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
# FIG C: graph representation and error response per sensor pooled across windows: axes separately
fig,axs=plt.subplots(1,3,figsize=(7.0,2.55),constrained_layout=True)
for ax,pct,c in zip(axs,[5,10,20],['#5E87AD','#6F9E96','#C98B65']):
 g=pts[pts.graph_edit_percent==pct]
 for seed,marker in [(7,'o'),(17,'^'),(27,'s')]:
  k=g[g.checkpoint_seed==seed]
  ax.scatter(k.response,k.mean_mae_change_mph,s=7,alpha=.32,c=c,marker=marker,edgecolors='none')
 ax.axhline(0,color='#7C8187',lw=.75,ls='--');ax.grid(alpha=.18)
 ax.set_title(f'{pct}% graph edit');ax.set_xlabel('Mean relative representation response')
axs[0].set_ylabel('Mean MAE change (mph)')
fig.savefig(FIG/'response_error.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
manifest={'run':'R19-reanalysis','derived_from':[(base/x).relative_to(ROOT).as_posix() for x in ['weak_to_strong.csv','calibration.csv','graph_input_seed7.npz','graph_input_seed17.npz','graph_input_seed27.npz']],'analysis_logic':'per-window-sensor horizon average valid error difference; relative Euclidean representation response','output_paths':{},'analysis_no_new_training':True}
for f in [*OUT.glob('*.csv'),*OUT.glob('*.json'),*FIG.glob('*.pdf')]:
 manifest['output_paths'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(agg,indent=2)); print(units.to_string(index=False));print('Figure3=',[str(x) for x in FIG.glob('*.pdf')])
