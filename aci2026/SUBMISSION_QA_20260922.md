# Submission QA — 22 September 2026

Target manuscript: Clinical Relevance Over Statistical Association: An Interpretable Probabilistic Layer for Prioritizing Drug-Drug Interaction Alerts
Target journal: Applied Clinical Informatics

## Public reproducibility assets checked before resubmission

The submission-specific assets are in aci2026/.

Checks performed on 22 September 2026:

- eval_harness.py compiled successfully with Python after installing the declared requirements.
- metrics_summary.csv was read successfully and matches the manuscript's frozen aggregate evaluation values.
- severe_validation_summary.csv was read successfully and matches the manuscript's reported severe-interaction enrichment summary.
- The repository README explicitly distinguishes public aggregate outputs from licensed DrugBank/UMLS/commercial-reference source material.

Key frozen values confirmed from the public files:

- drug-blind ROC-AUC: 0.763
- pair-blind ROC-AUC: 0.7615
- PR-AUC: 0.0503
- prevalence: 0.0192
- raw Brier score: 0.205
- calibrated Brier score: 0.0184
- top-1% severe-pair capture: 24%
- top-1% enrichment: 23.6x
- held-out severe pairs: 60% within top 25%

## Interpretation boundary

The manuscript evaluates an interpretable pre-context triage layer. It does not claim prospective reduction in alert overrides, adverse events, or patient-specific harm. Those outcomes require deployment studies with patient and workflow context.

The synthetic smoke-test path in eval_harness.py is provided only to verify code execution and must not be reported as scientific evidence.