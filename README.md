
# VeSCal

**Calibrated Multimodel Fusion for Fine-Grained Vegetation Community Classification**

**Authors:** Dristi Datta, Md Khalid Hasan Sakib, Manoranjan Paul, and Davina White

**Manuscript status:** Research manuscript prepared for submission to *Remote Sensing Applications: Society and Environment* (RSASE). This repository does not imply journal acceptance or publication.

**Earlier preprint:** [arXiv:2607.24160](https://arxiv.org/abs/2607.24160), *Calibrated Tree-Neural Fusion for Fine-Grained Vegetation Community Classification*. The arXiv version has an earlier three-author list and reports earlier experiments. It should **not** be treated as an identical record of the current four-author manuscript, corrected evaluation, or reported results.

## Overview

VeSCal (**Vegetation Stacking and Calibration**) is a tree--neural probability-fusion framework for fine-grained vegetation and associated land-cover classification. It combines six tree-based probability sources—Random Forest, XGBoost, ExtraTrees, histogram-based gradient boosting, LightGBM, and CatBoost—with a modality-aware neural branch, **EcoFuseNet-V2**. The evaluated procedure uses leakage-controlled out-of-fold (OOF) learning, validation-selected stacking or weighted soft voting, and post-hoc scalar temperature scaling.

The study uses **1,833 point-reference observations** covering **29 vegetation and non-vegetation classes** in Colorado's East River watershed. Its eight point-sampled environmental predictors comprise **six LiDAR-derived terrain and canopy variables** (elevation, slope, curvature, topographic position index, solar insolation, and canopy height) and **two hyperspectral-derived vegetation indices** (NDVI and NDWI). The analysis does **not** train on raw LiDAR point clouds, full hyperspectral image cubes, or the separately distributed vegetation classification map.

The publication name **VeSCal** refers to the same evaluated pipeline identified as `EcoTreeFuseNet-Plus` in the frozen implementation and archived artifacts. It does not indicate a separately retrained model.

## Release scope and reproducibility

This is a **lightweight source-code and processed-data release**. It includes documented code, the processed point-based reference table, and saved evaluation split indices. It deliberately **does not include trained checkpoints, frozen prediction arrays, full model-output archives, manuscript results tables, or results figures**.

Consequently, the reported statistical comparisons **cannot be independently recalculated from frozen predictions using this repository alone**. The included software self-tests are smoke tests, not complete reproduction of the manuscript experiments. See [Reproducibility notes](reproducibility/REPRODUCIBILITY.md) for the release manifest, checksums, and limitations.

The saved split indices are the authoritative records of the reported within-region and five-fold geographic evaluations. Regenerating split memberships with a different Python/scikit-learn environment did not reproduce the exact archived partitions; identical end-to-end retraining in a new environment has not been demonstrated. The dependency files give minimum declared requirements rather than a full lockfile for the original training environment.

## Repository contents

- **`vescal/`** — Training and model components, standalone baselines, probability calibration, and statistical-analysis code.
- **`data/`** — Processed point-reference table and source-data attribution.
- **`splits/`** — Saved train/validation/test membership arrays for the within-region evaluation and five geographic holdouts.
- **`scripts/check_release.py`** — Checks packaged file hashes and essential reference-data/split properties.
- **`reproducibility/`** — Reproducibility notes, manifest, checksums, traceability, and documented limitations.
- **`requirements.txt`** — Minimum declared training dependencies; not a complete historical environment lockfile.
- **`requirements-inference.txt`** — Dependencies declared for statistical-analysis tools.

## Installation and basic checks

Use an isolated Python environment. From the repository root, run:

```bash
python -m pip install -r requirements.txt
python vescal/test_training.py
python vescal/test_inference.py
python scripts/check_release.py
```

The two self-tests verify selected software functionality. Passing them does **not** reproduce the manuscript's full model training, geographic-holdout evaluation, or paired statistical findings.

## Optional training

```bash
python vescal/run_experiments.py --data-path data/vegetation_reference.csv --output-dir new_training_outputs --protocol both
```

Full training can be computationally intensive and may require environment-specific adjustments. This runner does **not automatically replay every archived geographic split as immutable input**. Use `splits/` to inspect the authoritative assignments used in the manuscript; do not assume a fresh run produces the same model parameters, split memberships, or numerical results.

For interpretation of preprocessing in the corrected pipeline, the VeSCal **tree probability branches use canopy-transformed predictors without feature standardization**. Standalone calibrated tree baselines and matched ExtraTrees predictor-group comparisons use standardization fitted on their training partitions. Neural OOF feature scalers are fitted only on the corresponding inner fitting subsets. These procedures should not be conflated when comparing implementations.

The statistical-analysis program `vescal/analyze_results.py` requires the full frozen model-output predictions, **which are not distributed in this lightweight release**. It therefore cannot reproduce the manuscript's statistical tables as a standalone command from the present repository.

## Experimental evaluation and main findings

Two evaluation designs were used:

1. **Within-region reference evaluation:** 1,273 training, 190 validation, and 370 test observations. Geographic blocks could contribute observations to more than one partition, so this measures within-region prediction rather than geographic transfer.
2. **Complete-block geographic evaluation:** 25 geographic blocks allocated to five nonoverlapping test folds containing 344, 253, 344, 369, and 523 observations. Each of the 1,833 observations is tested once. Training, validation, and test blocks are disjoint within each outer fold, but no spatial exclusion buffer is used.

Key **manuscript-reported** findings include:

- **Within-region calibrated VeSCal:** Accuracy **0.7973** and macro F1 **0.7782** on 370 test observations.
- **Geographic calibrated VeSCal:** Unweighted five-fold mean macro F1 **0.2555 ± 0.0686**.
- **Comparison with calibrated tree baselines:** Observed macro-F1 advantages were **not statistically significant** after Holm adjustment in either evaluation setting.
- **Geographic probability quality:** VeSCal had **significantly lower pooled multiclass Brier scores and negative log-likelihood (NLL)** than calibrated ExtraTrees, Random Forest, and XGBoost after Holm adjustment.
- **Matched ExtraTrees predictor comparison:** The full eight-variable set significantly improved pooled geographic macro F1 and Brier score over either the six-variable terrain-and-canopy set or the two spectral indices alone (Holm-adjusted **p = 0.0004** for all four contrasts).

The geographic macro-F1 value above is a **mean of fold-specific results**, not the fixed-29-class macro F1 computed from pooled geographic predictions for paired statistical inference. Similarly, improvements in Brier score and NLL should **not** be described as proof of uniformly better confidence calibration: geographic expected calibration error (ECE) contrasts between calibrated models were not Holm-significant.

The weighted-voting strategy was selected for all five geographic evaluations, while an elastic-net meta-classifier was selected for the within-region evaluation. No matched **tree-only fusion ablation** was available; therefore, the probability-score improvements cannot be attributed independently to the neural branch, its gates, embeddings, or prototype regularization.

## Interpretation and limitations

- Geographic-block holdout produced substantially lower fine-class discrimination than within-region testing. The two settings also differed in test composition and training-class coverage, so their difference is **not** a direct estimate of spatial leakage.
- Across the geographic test folds, **63 observations** belonged to classes absent from their respective training sets. These cases remain in the primary analysis; a supported-only evaluation is reported as a sensitivity analysis.
- Some supplied reference labels were assigned partly using environmental characteristics overlapping with the predictors, including elevation and NDVI. This creates potential **predictor–target dependence** for affected classes; independently field-verified labels were not available for external validation.
- All observations come from **one watershed and one airborne acquisition campaign**. The results do not establish transfer to independent watersheds, other acquisition periods, or new sensor configurations.
- The matched predictor-set comparisons support **complementary predictive information**, not isolated variable interactions or causal ecological relationships. No downstream conservation-management or field-sampling benefit was experimentally established.

## Data attribution and use

The environmental data and reference information originate from:

**Falco et al.** *Vegetation classification map and covariates associated with NEON AOP survey, East River, CO 2018*. ESS-DIVE dataset: https://doi.org/10.15485/1602034.

The included `data/vegetation_reference.csv` is a **processed, point-based analysis table containing 1,833 observations**. The data's original provenance, limitations, and applicable permissions remain governed by the source repository and its terms. Before redistribution, verify the dataset's reuse and attribution conditions, any restrictions concerning location-sensitive information, and approval from the manuscript's coauthors. This repository does not grant additional rights over the source data.

## Citation, authorship, and licensing

For the corrected results described here, cite the **current four-author manuscript** once a public, citable version becomes available, as well as the original ESS-DIVE dataset. See `CITATION.cff` for the repository's citation metadata; it should be checked against the current manuscript before release.

The earlier [arXiv preprint](https://arxiv.org/abs/2607.24160) remains a historical version with different authorship and experimental results. It should not be cited as though it contains the complete corrected analysis described here.

Software licensing remains subject to author approval; see `LICENSE_NOTICE.md`. If source-data redistribution permissions or coauthor approval are still pending, restrict public distribution of affected material until they are resolved. No new license or reuse permission is implied by this README.
