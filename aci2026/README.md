# ACI 2026 reproducibility assets

These files support the Applied Clinical Informatics manuscript:

**Clinical Relevance Over Statistical Association: An Interpretable Probabilistic Layer for Prioritizing Drug-Drug Interaction Alerts**

## What is public here

- `eval_harness.py` — leakage-aware drug-disjoint evaluation harness with ROC-AUC, PR-AUC, Brier score, expected calibration error, bootstrap confidence intervals, PU sensitivity, and synthetic smoke test.
- `metrics_summary.csv` — aggregate real-data evaluation metrics reported in the manuscript.
- `operating_points.csv` — threshold-versus-recall operating points.
- `pu_sensitivity.csv` — negative-ratio sensitivity results.
- `severe_validation_summary.csv` — aggregate severe-interaction enrichment summary.
- `goldset_characterization.csv` — structural characterization of the 21,897-pair reference set.
- `mechanism_distribution.csv` — mechanism-class support counts.

## What is intentionally not redistributed

The full continuous nine-feature matrix and some pair-level source material depend on licensed DrugBank/UMLS or commercial-reference content. Those source-level materials are not redistributed here. The manuscript and supplement therefore distinguish reproducible public methodology and aggregate outputs from licensed source data.

The pair-level severe-interaction crosswalk used for a secondary validation is also not mirrored here because it contains source-derived clinical-reference information. The public `severe_validation_summary.csv` reports only aggregate results.

## Important interpretation boundary

The study evaluates an **interpretable pre-context triage layer** for DDI alerts. It does not claim prospective reduction in clinician overrides, adverse drug events, or patient-specific harm. Those require deployment studies with dose, timing, comorbidity, laboratory values, and workflow context.

## Running the harness

Install NumPy, pandas, scikit-learn, SciPy, and Matplotlib, then run:

```bash
python eval_harness.py --smoke
```

The smoke-test data are synthetic and must not be reported as scientific results. To reproduce the real-data analysis, authorized users must supply the licensed feature matrix in the format documented inside `eval_harness.py`.

## Submission integrity

The manuscript reports the aggregate values frozen in this directory. No GitHub content should be interpreted as independent clinical validation or as a prescribing recommendation.
