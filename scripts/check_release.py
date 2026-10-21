#!/usr/bin/env python3
"""Verify repository manifest and basic data/partition invariants."""
import csv,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    with (ROOT/'reproducibility/file_manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file() or p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:
            raise RuntimeError(f'File manifest mismatch: {row["path"]}')
    data=pd.read_csv(ROOT/'data/vegetation_reference.csv')
    assert len(data)==1833 and data['class'].nunique()==29
    folders=[ROOT/'splits/within_region']+sorted((ROOT/'splits/geographic_holdout').glob('fold_*'))
    assert len(folders)==6
    for folder in folders:
        arrays=[np.load(folder/f'{name}_indices.npy') for name in ('train','validation','test')]
        combined=np.concatenate(arrays)
        assert len(combined)==1833 and len(np.unique(combined))==1833
        assert combined.min()>=0 and combined.max()<1833
    assert sum(len(np.load(p/'test_indices.npy')) for p in folders[1:])==1833
    print(f'VeSCal release integrity verified: {len(rows)} files, 6 partitions, 1833 observations, 29 classes.')
if __name__=='__main__':main()
