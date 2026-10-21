from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def forest_plot(df: pd.DataFrame, out: str):
    d=df[(df['metric']=='macro_f1') & (df['scope'].isin(['primary','spatial']))].copy()
    if d.empty: return
    d['label']=d['scope']+': '+d['model_a']+' vs '+d['model_b']
    y=np.arange(len(d)); x=d['delta'].to_numpy(); lo=x-d['ci_low'].to_numpy(); hi=d['ci_high'].to_numpy()-x
    fig,ax=plt.subplots(figsize=(8,max(3,0.45*len(d)+1.5))); ax.errorbar(x,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=3)
    ax.axvline(0,linewidth=1); ax.set_yticks(y); ax.set_yticklabels(d['label']); ax.set_xlabel('Paired difference in Macro-F1 (A − B)'); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out,dpi=300); plt.close(fig)

def spatial_fold_plot(fold_df: pd.DataFrame, out: str):
    d=fold_df[(fold_df['state']=='calibrated') & (fold_df['model'].isin(['proposed_fusion','extra_trees','rf','xgboost']))]
    if d.empty:return
    p=d.pivot(index='spatial_fold',columns='model',values='macro_f1').sort_index()
    ax=p.plot(marker='o',figsize=(8,4.5)); ax.set_xlabel('Strict spatial fold'); ax.set_ylabel('Macro-F1'); ax.set_title('Strict spatial block holdout'); ax.get_figure().tight_layout(); ax.get_figure().savefig(out,dpi=300); plt.close(ax.get_figure())

def modality_plot(mod_df: pd.DataFrame, out: str):
    d=mod_df[(mod_df['split_mode']=='spatial_block') & (mod_df['state']=='raw')]
    if d.empty:return
    g=d.groupby('modality')['macro_f1'].agg(['mean','std']).reindex(['lidar_derived','spectral','fused'])
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(g.index,g['mean'],yerr=g['std'],capsize=4); ax.set_ylabel('Macro-F1 (mean ± sample SD)'); ax.set_xlabel('ExtraTrees input group'); fig.tight_layout(); fig.savefig(out,dpi=300); plt.close(fig)
