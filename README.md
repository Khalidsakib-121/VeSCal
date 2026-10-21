# VeSCal

**Calibrated Multimodel Fusion for Fine-Grained Vegetation Community Mapping**

VeSCal (Vegetation Stacking and Calibration) combines six tree ensembles and a modality-aware neural network through leakage-controlled out-of-fold learning, validation-selected stacking or weighted voting, and post-hoc temperature scaling. The accompanying study uses 1,833 labelled observations representing 29 vegetation and non-vegetation classes in the East River watershed, Colorado, with LiDAR-derived terrain/canopy predictors and two hyperspectral-derived vegetation indices.

This is the **lightweight source-code and data release** accompanying a manuscript prepared for *Remote Sensing of Environment*. It deliberately **does not include trained checkpoints, frozen prediction arrays, model output archives, results tables, or result figures**. Published summary statistics cannot be independently recalculated from frozen predictions using only this repository; see [Reproducibility notes](reproducibility/REPRODUCIBILITY.md).

## Repository contents

| Path | Purpose |
| --- | --- |
| `vescal/` | Training, model components, baseline comparisons, calibration, and statistical-analysis code |
| `data/` | Point-based reference table and data attribution |
| `splits/` | Saved train/validation/test membership arrays for within-region and five geographic holdouts |
| `scripts/check_release.py` | Check packaged file hashes and essential reference-data/split properties |
| `reproducibility/` | Traceability, file manifest, checksums, limitations and documentation |
| `requirements.txt` | Minimum declared training dependencies (not a full historical lockfile) |
| `requirements-inference.txt` | Dependencies recorded for the statistical analysis tools |

## Installation and basic checks

Use an isolated Python environment and install the packages required for training:

```bash
python -m pip install -r requirements.txt
python vescal/test_training.py
python vescal/test_inference.py
python scripts/check_release.py
```

The two self-tests are software smoke tests; they do not reproduce the manuscript's full model-training experiments or statistical findings.

## Optional training

```bash
python vescal/run_experiments.py --data-path data/vegetation_reference.csv --output-dir new_training_outputs --protocol both
```

Full training is computationally intensive and may require further environment-specific adjustments. This runner **does not automatically replay every archived geographic split as immutable input**. The saved index arrays in `splits/` document the authoritative membership of the original evaluation partitions. Identical end-to-end retraining under a new software environment has not been demonstrated.

The included `vescal/analyze_results.py` statistical program expects full frozen model-output predictions, **which are not distributed in this lightweight repository**. It therefore cannot reproduce the published statistical comparisons as a standalone command here. Readers should not confuse software smoke tests with a complete reproducibility claim.

## Data attribution and use

Original source: Falco et al., *Vegetation classification map and covariates associated with NEON AOP survey, East River, CO 2018*, ESS-DIVE: https://doi.org/10.15485/1602034.

The included `data/vegetation_reference.csv` is a processed point-based analysis table (1,833 observations). Before public release, verify the source dataset's reuse/attribution conditions and any location-sensitive redistribution considerations. This repository does not assign new usage rights to the source data.

## Scientific scope

The reported within-region macro F1 for calibrated VeSCal is 0.7782, while the unweighted five-fold geographic-holdout mean is 0.2555. The manuscript reports significant geographic Brier and NLL improvements over calibrated tree baselines, but not significant macro-F1 differences after Holm adjustment. These are **manuscript-reported results**, not independently reproduced by the present slim distribution.

This repository does not establish generalization to independent watersheds or isolate the incremental contribution of the neural branch.

## Citation and licensing

Please cite both the associated manuscript, when available, and the source dataset. See `CITATION.cff`. Software licensing remains pending author approval; see `LICENSE_NOTICE.md`. Keep the repository private until source-data redistribution permissions and coauthor approval have been confirmed.
