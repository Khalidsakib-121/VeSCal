from __future__ import annotations
import argparse, json, shutil, zlib
from pathlib import Path
import numpy as np, pandas as pd
from inference import *
from plots import forest_plot, spatial_fold_plot, modality_plot

PRIMARY_COMPARATORS=['extra_trees','rf','xgboost']
PRIMARY_METRICS=['macro_f1','brier','nll','ece']

def stable_offset(*parts):
    return zlib.crc32('::'.join(map(str, parts)).encode('utf-8')) % 10000

def ensure_alignment(root: Path):
    primary=load_run(root/'primary_seed42')
    folds=[load_run(root/f'spatial_fold_{i}') for i in range(5)]
    idx=np.concatenate([f['test_indices'] for f in folds]); assert len(idx)==1833 and len(np.unique(idx))==1833 and set(idx.tolist())==set(range(1833))
    ids=np.concatenate([f['test_sample_ids'] for f in folds]); assert len(np.unique(ids))==1833
    for f in folds:
        assert len(set(f['train_indices']).intersection(set(f['test_indices'])))==0
        assert len(set(f['validation_indices']).intersection(set(f['test_indices'])))==0
    return primary,folds

def main():
    ap=argparse.ArgumentParser(description='Phase 4: statistical testing and evidence-based assessment of frozen Phase 3 outputs')
    ap.add_argument('--phase3',required=True,help='Phase_3_FREEZE_OUTPUTS.zip or extracted outputs_phase3_frozen directory')
    ap.add_argument('--output',default='outputs_phase4')
    ap.add_argument('--resamples',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=4204)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    root,tmp=locate_phase3_root(a.phase3)
    try:
        primary,folds=ensure_alignment(root); K=primary['K']
        # Freeze input summaries for traceability
        for fn in ['phase3_all_main_comparisons.csv','phase3_all_modality_comparisons.csv','phase3_spatial_main_summary.csv','phase3_spatial_modality_summary.csv']:
            shutil.copy2(root/fn,out/f'INPUT_{fn}')
        comparisons=[]; mcn=[]
        # Primary mixed split: calibrated fusion vs equally calibrated strong trees
        y=primary['y']; pa=primary['models']['proposed_fusion']['calibrated']
        for comp in PRIMARY_COMPARATORS:
            pb=primary['models'][comp]['calibrated']
            for metric in PRIMARY_METRICS:
                r=paired_bootstrap(y,pa,pb,K,metric,B=a.resamples,seed=a.seed+stable_offset(comp,metric))
                pr=paired_randomization(y,pa,pb,K,metric,B=a.resamples,seed=a.seed+10000+stable_offset(comp,metric))
                comparisons.append(dict(scope='primary',family=f'primary_{metric}',model_a='proposed_fusion_calibrated',model_b=f'{comp}_calibrated',metric=metric,**r,p_randomization=pr,resampling_unit='observation',macro_definition='Phase3 sklearn macro over represented union'))
            mt=mcnemar_test(y,pa,pb); mcn.append(dict(scope='primary',model_a='proposed_fusion_calibrated',model_b=f'{comp}_calibrated',**mt))
        # Calibration effect within each model on primary
        for model in ['proposed_fusion','extra_trees','rf','xgboost']:
            cal=primary['models'][model]['calibrated']; raw=primary['models'][model]['raw']
            for metric in ['brier','nll','ece']:
                # lower is better; define improvement = raw - calibrated, so positive favors calibration
                r=paired_bootstrap(y,raw,cal,K,metric,B=a.resamples,seed=a.seed+20000+stable_offset(model,metric))
                pr=paired_randomization(y,raw,cal,K,metric,B=a.resamples,seed=a.seed+30000+stable_offset(model,metric))
                comparisons.append(dict(scope='primary_calibration',family=f'primary_calibration_{metric}',model_a=f'{model}_raw',model_b=f'{model}_calibrated',metric=f'{metric}_raw_minus_calibrated',**r,p_randomization=pr,resampling_unit='observation',macro_definition='NA'))
        # Spatial pooled OOF in original-sample order, with physical block as cluster
        order=np.argsort(np.concatenate([f['test_indices'] for f in folds])); ysp=np.concatenate([f['y'] for f in folds])[order]
        blocks=np.concatenate([f['test_blocks'] for f in folds])[order]
        pool={m:{s:np.concatenate([f['models'][m][s] for f in folds],axis=0)[order] for s in ['raw','calibrated']} for m in ['proposed_fusion','extra_trees','rf','xgboost']}
        assert len(np.unique(blocks))>=20
        for comp in PRIMARY_COMPARATORS:
            pa=pool['proposed_fusion']['calibrated']; pb=pool[comp]['calibrated']
            for metric in PRIMARY_METRICS:
                # Fixed-K Macro-F1 during block resampling prevents denominator changes when a resample omits classes.
                fixed=(metric=='macro_f1')
                r=cluster_bootstrap(ysp,pa,pb,blocks,K,metric,B=a.resamples,seed=a.seed+40000+stable_offset(comp,metric),fixed29=fixed)
                pr=paired_randomization(ysp,pa,pb,K,metric,B=a.resamples,seed=a.seed+50000+stable_offset(comp,metric),units=blocks,fixed29=fixed)
                comparisons.append(dict(scope='spatial',family=f'spatial_{metric}',model_a='proposed_fusion_calibrated',model_b=f'{comp}_calibrated',metric=metric,**r,p_randomization=pr,resampling_unit='geographic_block',macro_definition='fixed K=29 for block-resampling inference' if fixed else 'NA'))
        # Supported-only spatial sensitivity: remove observations whose true class was absent from that fold's training set.
        support_parts=[]
        for i,f in enumerate(folds):
            sa=load_json(root/f'spatial_fold_{i}'/'phase3_class_support_audit.json')
            absent=np.asarray(sa.get('test_classes_absent_from_train',[]),dtype=int)
            support_parts.append(~np.isin(f['y'],absent))
        support_mask=np.concatenate(support_parts)[order]
        for comp in PRIMARY_COMPARATORS:
            pa=pool['proposed_fusion']['calibrated'][support_mask]; pb=pool[comp]['calibrated'][support_mask]
            ys=ysp[support_mask]; bs=blocks[support_mask]
            r=cluster_bootstrap(ys,pa,pb,bs,K,'macro_f1',B=a.resamples,seed=a.seed+55000+stable_offset(comp,'supported'),fixed29=True)
            pr=paired_randomization(ys,pa,pb,K,'macro_f1',B=a.resamples,seed=a.seed+56000+stable_offset(comp,'supported'),units=bs,fixed29=True)
            comparisons.append(dict(scope='spatial_supported_only',family='spatial_supported_only_macro_f1',model_a='proposed_fusion_calibrated',model_b=f'{comp}_calibrated',metric='macro_f1',**r,p_randomization=pr,resampling_unit='geographic_block',macro_definition='fixed K=29; unsupported true-class observations removed identically for all models'))

        # Spatial calibration transfer within each model
        for model in ['proposed_fusion','extra_trees','rf','xgboost']:
            raw=pool[model]['raw']; cal=pool[model]['calibrated']
            for metric in ['brier','nll','ece']:
                r=cluster_bootstrap(ysp,raw,cal,blocks,K,metric,B=a.resamples,seed=a.seed+60000+stable_offset(model,metric),fixed29=False)
                pr=paired_randomization(ysp,raw,cal,K,metric,B=a.resamples,seed=a.seed+70000+stable_offset(model,metric),units=blocks)
                comparisons.append(dict(scope='spatial_calibration',family=f'spatial_calibration_{metric}',model_a=f'{model}_raw',model_b=f'{model}_calibrated',metric=f'{metric}_raw_minus_calibrated',**r,p_randomization=pr,resampling_unit='geographic_block',macro_definition='NA'))
        # Fair modality: ExtraTrees, strict spatial, fused vs each restricted input. Raw for discrimination, calibrated for probability metrics.
        modpool={mod:{s:np.concatenate([f['modalities'][mod][s] for f in folds],axis=0)[order] for s in ['raw','calibrated']} for mod in ['fused','lidar_derived','spectral']}
        for other in ['lidar_derived','spectral']:
            for metric in ['macro_f1','brier','nll','ece']:
                state='raw' if metric=='macro_f1' else 'calibrated'; fixed=(metric=='macro_f1')
                r=cluster_bootstrap(ysp,modpool['fused'][state],modpool[other][state],blocks,K,metric,B=a.resamples,seed=a.seed+80000+stable_offset(other,metric),fixed29=fixed)
                pr=paired_randomization(ysp,modpool['fused'][state],modpool[other][state],K,metric,B=a.resamples,seed=a.seed+90000+stable_offset(other,metric),units=blocks,fixed29=fixed)
                comparisons.append(dict(scope='spatial_modality',family=f'spatial_modality_{metric}',model_a=f'extra_trees_fused_{state}',model_b=f'extra_trees_{other}_{state}',metric=metric,**r,p_randomization=pr,resampling_unit='geographic_block',macro_definition='fixed K=29 for block-resampling inference' if fixed else 'NA'))
        comp=pd.DataFrame(comparisons); comp=add_holm_by_family(comp)
        comp['better_direction']=comp['metric'].map(lambda x: 'positive favors A' if ('macro_f1' in str(x) or 'raw_minus_calibrated' in str(x)) else ('negative favors A' if str(x) in {'brier','nll','ece'} else 'see metric definition'))
        comp['holm_significant_0_05']=comp['p_holm']<0.05
        comp.to_csv(out/'phase4_statistical_comparisons.csv',index=False)
        pd.DataFrame(mcn).to_csv(out/'phase4_mcnemar_primary.csv',index=False)
        # Descriptive pooled spatial metrics + fold results + support
        rows=[]
        for model in pool:
            for state in ['raw','calibrated']:
                mb=metric_bundle(ysp,pool[model][state],K); rows.append(dict(model=model,state=state,n=len(ysp),n_blocks=len(np.unique(blocks)),**mb))
        pooled_df=pd.DataFrame(rows); pooled_df.to_csv(out/'phase4_spatial_pooled_metrics.csv',index=False)
        # Supported-only pooled metrics, common support mask for every comparator.
        srows=[]
        for model in pool:
            for state in ['raw','calibrated']:
                mb=metric_bundle(ysp[support_mask],pool[model][state][support_mask],K); srows.append(dict(model=model,state=state,n=int(support_mask.sum()),n_blocks=int(len(np.unique(blocks[support_mask]))),**mb))
        pd.DataFrame(srows).to_csv(out/'phase4_spatial_supported_only_pooled_metrics.csv',index=False)
        fold_df=pd.read_csv(root/'phase3_all_main_comparisons.csv'); fold_df=fold_df[fold_df['split_mode']=='spatial_block']; fold_df.to_csv(out/'phase4_spatial_fold_metrics.csv',index=False)
        mod_df=pd.read_csv(root/'phase3_all_modality_comparisons.csv'); mod_df.to_csv(out/'phase4_modality_all_runs.csv',index=False)
        # Fold-by-fold calibration transfer (positive improvement means the lower-is-better metric decreased after calibration).
        cal_rows=[]
        for i,f in enumerate(folds):
            for model in ['proposed_fusion','extra_trees','rf','xgboost']:
                for metric in ['ece','brier','nll']:
                    raw=metric_value(metric,f['y'],f['models'][model]['raw'],K); cal=metric_value(metric,f['y'],f['models'][model]['calibrated'],K)
                    cal_rows.append(dict(spatial_fold=i,model=model,metric=metric,raw=raw,calibrated=cal,improvement_raw_minus_calibrated=raw-cal,improved=bool(raw-cal>0)))
        pd.DataFrame(cal_rows).to_csv(out/'phase4_calibration_transfer_by_fold.csv',index=False)
        # Support report
        supp=[]
        for i,f in enumerate(folds):
            audit=load_json(root/f'spatial_fold_{i}'/'phase3_class_support_audit.json'); supp.append({'spatial_fold':i,**audit})
        with open(out/'phase4_class_support.json','w',encoding='utf-8') as fh: json.dump(supp,fh,indent=2,default=str)
        # Claim/evidence matrix from actual tests
        def get(scope,fam,a_model,b_model,metric):
            q=comp[(comp.scope==scope)&(comp.family==fam)&(comp.model_a==a_model)&(comp.model_b==b_model)]
            return q.iloc[0] if len(q) else None
        claims=[]
        for comparator in PRIMARY_COMPARATORS:
            r=get('primary','primary_macro_f1','proposed_fusion_calibrated',f'{comparator}_calibrated','macro_f1')
            claims.append({'claim':f'Primary-split discrimination advantage over calibrated {comparator}','status':'SUPPORTED' if r.p_holm<0.05 and r.delta>0 else 'QUALIFIED / NOT STATISTICALLY SUPPORTED','evidence':f"ΔMacro-F1={r.delta:.6f}; 95% CI [{r.ci_low:.6f}, {r.ci_high:.6f}]; Holm p={r.p_holm:.6g}"})
        for comparator in PRIMARY_COMPARATORS:
            r=get('spatial','spatial_macro_f1','proposed_fusion_calibrated',f'{comparator}_calibrated','macro_f1')
            claims.append({'claim':f'Strict-spatial discrimination advantage over calibrated {comparator}','status':'SUPPORTED' if r.p_holm<0.05 and r.delta>0 else 'QUALIFIED / NOT STATISTICALLY SUPPORTED','evidence':f"Block-resampled ΔMacro-F1={r.delta:.6f}; 95% CI [{r.ci_low:.6f}, {r.ci_high:.6f}]; Holm p={r.p_holm:.6g}"})
        for other in ['lidar_derived','spectral']:
            r=get('spatial_modality','spatial_modality_macro_f1','extra_trees_fused_raw',f'extra_trees_{other}_raw','macro_f1')
            claims.append({'claim':f'Combined eight predictors outperform {other} under the same ExtraTrees learner in strict spatial evaluation','status':'SUPPORTED' if r.p_holm<0.05 and r.delta>0 else 'QUALIFIED / NOT STATISTICALLY SUPPORTED','evidence':f"Block-resampled ΔMacro-F1={r.delta:.6f}; 95% CI [{r.ci_low:.6f}, {r.ci_high:.6f}]; Holm p={r.p_holm:.6g}"})
        claims.append({'claim':'Latent neural embeddings/gates/prototype distances materially improve final fusion','status':'UNRESOLVED — PHASE 3 DOES NOT CONTAIN A MATCHED TREE-ONLY STACK ABLATION','evidence':'Do not infer marginal benefit from coefficient magnitude alone.'})
        claims.append({'claim':'Calibration reliably transfers to unseen geographic blocks for every model','status':'MUST BE ASSESSED MODEL-BY-MODEL; UNIVERSAL CLAIM NOT PERMITTED','evidence':'See phase4_statistical_comparisons.csv spatial_calibration families.'})
        pd.DataFrame(claims).to_csv(out/'phase4_claim_evidence_matrix.csv',index=False)
        # Figures
        forest_plot(comp,out/'figures'/'phase4_macro_f1_forest.png'); spatial_fold_plot(fold_df,out/'figures'/'phase4_spatial_folds.png'); modality_plot(mod_df,out/'figures'/'phase4_modality_spatial.png')
        # Audit
        audit={'phase':'4','phase3_input':str(Path(a.phase3).resolve()),'phase3_root':str(root),'resamples':a.resamples,'seed':a.seed,'n_spatial_blocks':int(len(np.unique(blocks))),'spatial_test_samples':int(len(ysp)),'K':K,'spatial_test_partition_exact_cover':True,'statistical_design':{'primary':'paired observation bootstrap + paired row randomization; McNemar hard-label tests','spatial':'geographic-block bootstrap + block-level paired randomization','multiplicity':'Holm within prespecified metric/scope families','spatial_macro_resampling':'fixed K=29 denominator to prevent class-denominator drift'},'note':'Phase 4 analyzes frozen Phase 3 predictions only; it performs no refitting or model selection.'}
        # Input and output integrity manifest.
        audit['phase3_input_sha256']=sha256(Path(a.phase3).resolve()) if Path(a.phase3).is_file() else None
        with open(out/'PHASE4_RUN_AUDIT.json','w',encoding='utf-8') as f: json.dump(audit,f,indent=2)
        manifest=[]
        for q in sorted(out.rglob('*')):
            if q.is_file(): manifest.append({'relative_path':str(q.relative_to(out)),'bytes':q.stat().st_size,'sha256':sha256(q)})
        pd.DataFrame(manifest).to_csv(out/'PHASE4_OUTPUT_MANIFEST.csv',index=False)
        print('PHASE 4 COMPLETE:',out.resolve())
    finally:
        if tmp is not None: tmp.cleanup()
if __name__=='__main__': main()
