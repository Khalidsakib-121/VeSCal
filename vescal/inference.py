from __future__ import annotations
import json, math, tempfile, zipfile, warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

EPS=1e-12


def locate_phase3_root(path: str, temp_parent: str | None = None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    p=Path(path).resolve()
    if p.is_dir():
        if (p/'outputs_phase3_frozen').is_dir(): return p/'outputs_phase3_frozen', None
        if (p/'primary_seed42').is_dir(): return p, None
        raise FileNotFoundError(f'Cannot locate outputs_phase3_frozen under {p}')
    if p.is_file() and p.suffix.lower()=='.zip':
        td=tempfile.TemporaryDirectory(prefix='phase4_phase3_', dir=temp_parent)
        with zipfile.ZipFile(p,'r') as z: z.extractall(td.name)
        base=Path(td.name)
        candidates=[base/'outputs_phase3_frozen']+[q for q in base.rglob('outputs_phase3_frozen') if q.is_dir()]
        if not candidates or not candidates[0].is_dir():
            td.cleanup(); raise FileNotFoundError('ZIP does not contain outputs_phase3_frozen')
        return candidates[0], td
    raise FileNotFoundError(path)


def load_json(path: Path):
    with open(path,'r',encoding='utf-8') as f: return json.load(f)


def compute_ece(y_true, probs, n_bins=10):
    y_true=np.asarray(y_true,int); probs=np.asarray(probs,float)
    conf=probs.max(1); pred=probs.argmax(1); correct=(pred==y_true).astype(float)
    edges=np.linspace(0,1,n_bins+1); out=0.0
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>=lo)&((conf<=hi) if i==n_bins-1 else (conf<hi))
        if mask.any(): out += mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return float(out)

def compute_brier(y_true, probs, K=None):
    y_true=np.asarray(y_true,int); p=np.asarray(probs,float); K=K or p.shape[1]
    y=np.zeros_like(p); y[np.arange(len(y_true)),y_true]=1
    return float(np.mean(np.sum((p-y)**2,axis=1)))

def compute_nll(y_true, probs):
    y_true=np.asarray(y_true,int); p=np.clip(np.asarray(probs,float),1e-8,1.0)
    return float(-np.mean(np.log(p[np.arange(len(y_true)),y_true])))

def metric_value(name, y_true, probs, K, fixed29=False):
    p=np.asarray(probs,float); pred=p.argmax(1); y=np.asarray(y_true,int)
    if name=='accuracy': return float(np.mean(pred==y))
    if name=='macro_f1':
        if fixed29: return float(f1_score(y,pred,labels=list(range(K)),average='macro',zero_division=0))
        return float(f1_score(y,pred,average='macro',zero_division=0))
    if name=='balanced_accuracy':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true', category=UserWarning)
            return float(balanced_accuracy_score(y,pred))
    if name=='mcc': return float(matthews_corrcoef(y,pred))
    if name=='ece': return compute_ece(y,p)
    if name=='brier': return compute_brier(y,p,K)
    if name=='nll': return compute_nll(y,p)
    raise ValueError(name)

def metric_bundle(y_true, probs, K):
    return {m:metric_value(m,y_true,probs,K) for m in ['accuracy','macro_f1','balanced_accuracy','mcc','ece','brier','nll']} | {
        'macro_f1_fixedK': metric_value('macro_f1',y_true,probs,K,fixed29=True)
    }

MODEL_FILES={
 'rf': {'raw':'phase3_rf_test_probs_raw.npy','calibrated':'phase3_rf_test_probs_calibrated.npy'},
 'xgboost': {'raw':'phase3_xgboost_test_probs_raw.npy','calibrated':'phase3_xgboost_test_probs_calibrated.npy'},
 'extra_trees': {'raw':'phase3_extra_trees_test_probs_raw.npy','calibrated':'phase3_extra_trees_test_probs_calibrated.npy'},
 'proposed_fusion': {'raw':'probabilities_ecotreefusenet_plus.npy','calibrated':'probabilities_calibrated_ecotreefusenet_plus.npy'},
}
MODALITY_FILES={
 'fused': {'raw':'phase3_modality_fused_extra_trees_test_raw.npy','calibrated':'phase3_modality_fused_extra_trees_test_calibrated.npy'},
 'lidar_derived': {'raw':'phase3_modality_lidar_derived_extra_trees_test_raw.npy','calibrated':'phase3_modality_lidar_derived_extra_trees_test_calibrated.npy'},
 'spectral': {'raw':'phase3_modality_spectral_extra_trees_test_raw.npy','calibrated':'phase3_modality_spectral_extra_trees_test_calibrated.npy'},
}

def load_run(run_dir: Path) -> Dict:
    pred=run_dir/'predictions'; y=np.load(pred/'y_true.npy'); names=load_json(run_dir/'class_names.json'); K=len(names)
    out={'run_dir':run_dir,'y':y,'K':K,'class_names':names,'models':{},'modalities':{}}
    for model,states in MODEL_FILES.items():
        out['models'][model]={}
        for state,fn in states.items():
            p=np.load(pred/fn); assert p.shape==(len(y),K),(run_dir,model,state,p.shape,len(y),K)
            out['models'][model][state]=p
    for mod,states in MODALITY_FILES.items():
        out['modalities'][mod]={}
        for state,fn in states.items():
            p=np.load(pred/fn); assert p.shape==(len(y),K)
            out['modalities'][mod][state]=p
    for nm in ['test_indices.npy','test_sample_ids.npy','train_indices.npy','train_sample_ids.npy','validation_indices.npy','validation_sample_ids.npy']:
        q=run_dir/nm
        if q.exists(): out[nm[:-4]]=np.load(q,allow_pickle=True)
    blocks=np.load(run_dir/'all_spatial_block_ids.npy',allow_pickle=True)
    ti=out.get('test_indices')
    if ti is not None: out['test_blocks']=blocks[ti]
    else: out['test_blocks']=np.array(['NA']*len(y),object)
    return out


def paired_bootstrap(y, pa, pb, K, metric, B=5000, seed=4204, fixed29=False):
    rng=np.random.default_rng(seed); n=len(y); vals=np.empty(B,float)
    for b in range(B):
        idx=rng.integers(0,n,n)
        vals[b]=metric_value(metric,y[idx],pa[idx],K,fixed29)-metric_value(metric,y[idx],pb[idx],K,fixed29)
    obs=metric_value(metric,y,pa,K,fixed29)-metric_value(metric,y,pb,K,fixed29)
    lo,hi=np.quantile(vals,[.025,.975]); p=2*min((np.sum(vals<=0)+1)/(B+1),(np.sum(vals>=0)+1)/(B+1)); p=min(1.0,p)
    return dict(delta=obs,ci_low=float(lo),ci_high=float(hi),p_boot=float(p))


def paired_randomization(y, pa, pb, K, metric, B=5000, seed=4214, units=None, fixed29=False):
    rng=np.random.default_rng(seed); obs=metric_value(metric,y,pa,K,fixed29)-metric_value(metric,y,pb,K,fixed29)
    if units is None: units=np.arange(len(y))
    units=np.asarray(units); uniq=np.unique(units); vals=np.empty(B,float)
    for b in range(B):
        swap_units=set(uniq[rng.random(len(uniq))<.5].tolist()); mask=np.array([u in swap_units for u in units])
        xa=pa.copy(); xb=pb.copy(); xa[mask]=pb[mask]; xb[mask]=pa[mask]
        vals[b]=metric_value(metric,y,xa,K,fixed29)-metric_value(metric,y,xb,K,fixed29)
    p=(np.sum(np.abs(vals)>=abs(obs))+1)/(B+1)
    return float(p)


def cluster_bootstrap(y, pa, pb, units, K, metric, B=5000, seed=4224, fixed29=True):
    rng=np.random.default_rng(seed); units=np.asarray(units); uniq=np.unique(units); groups={u:np.flatnonzero(units==u) for u in uniq}
    vals=np.empty(B,float)
    for b in range(B):
        draw=rng.choice(uniq,size=len(uniq),replace=True); idx=np.concatenate([groups[u] for u in draw])
        vals[b]=metric_value(metric,y[idx],pa[idx],K,fixed29)-metric_value(metric,y[idx],pb[idx],K,fixed29)
    obs=metric_value(metric,y,pa,K,fixed29)-metric_value(metric,y,pb,K,fixed29)
    lo,hi=np.quantile(vals,[.025,.975]); p=2*min((np.sum(vals<=0)+1)/(B+1),(np.sum(vals>=0)+1)/(B+1)); p=min(1.0,p)
    return dict(delta=obs,ci_low=float(lo),ci_high=float(hi),p_boot=float(p),n_units=int(len(uniq)))


def mcnemar_test(y, pa, pb):
    a=pa.argmax(1)==y; b=pb.argmax(1)==y
    n01=int(np.sum(a & ~b)); n10=int(np.sum(~a & b)); d=n01+n10
    if d==0: return dict(n01=n01,n10=n10,p=1.0,method='no discordance')
    if d<25:
        p=float(stats.binomtest(min(n01,n10),n=d,p=0.5,alternative='two-sided').pvalue); method='exact binomial'
    else:
        chi=((abs(n01-n10)-1)**2)/d; p=float(stats.chi2.sf(chi,1)); method='continuity-corrected chi-square'
    return dict(n01=n01,n10=n10,p=p,method=method)


def holm_adjust(pvals: Iterable[float]) -> List[float]:
    p=np.asarray(list(pvals),float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj.tolist()


def add_holm_by_family(df: pd.DataFrame, p_col='p_randomization') -> pd.DataFrame:
    df=df.copy(); df['p_holm']=np.nan
    for fam,idx in df.groupby('family').groups.items():
        vals=df.loc[idx,p_col].astype(float).tolist(); df.loc[idx,'p_holm']=holm_adjust(vals)
    return df


def sha256(path: Path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
