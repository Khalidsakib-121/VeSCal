# Reproducibility and provenance

## Available materials

This code-only release contains the corrected VeSCal source, its 1,833-row prepared reference table, and saved numeric train/validation/test indices for one within-region split and five geographic holdouts. `reproducibility/file_manifest.csv` and `reproducibility/checksums.sha256` cover the files in this distribution. `reproducibility/source_mapping.csv` documents the source-to-release script correspondence.

## Deliberately excluded

To keep this GitHub repository lightweight, it **does not contain** the original frozen model-output archive, probability arrays, trained weights, archived predictions, full analysis outputs, or precomputed figures/tables. Therefore, the statistical analysis script cannot regenerate the manuscript's paired comparisons from frozen predictions using this checkout alone. Access to the frozen predictions or independent full training is needed for that analysis.

## Verification commands

```bash
python -m pip install -r requirements.txt
python vescal/test_training.py
python vescal/test_inference.py
python scripts/check_release.py
```

These commands check key software behaviour and file/data/split integrity; they do **not** validate the manuscript's statistical results from archived predictions.

## Training and evaluation boundaries

```bash
python vescal/run_experiments.py --data-path data/vegetation_reference.csv --output-dir new_training_outputs --protocol both
```

The trainer may reconstruct fold memberships using its split-generation procedure rather than consuming the immutable indices in `splits/`. In an earlier independent attempt under a different Python/scikit-learn environment, generated fold assignments differed from the archived originals. The archived arrays remain the authoritative records for the reported evaluation, and the provided command is not guaranteed to regenerate those exact folds.

A complete pinned record of the original execution environment is not available; the distributed requirements describe minimum package versions. No claim of independent full end-to-end retraining reproduction is made. The inclusion of executable analysis code is not a claim that the omitted frozen predictions are distributed.

## Reporting scope

The study reports statistically significant geographic Brier and NLL differences from three calibrated tree-ensemble baselines, but the macro-F1 comparisons do not remain significant after Holm adjustment. The 63 unsupported-class geographic test observations were retained in the primary evaluation. The present lightweight repository supplies supporting code and split metadata, not direct raw evidence for rechecking those numeric claims.
