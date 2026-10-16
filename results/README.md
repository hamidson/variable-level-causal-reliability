# Experimental Results

This directory contains the archived outputs used in the revised ACIJ manuscript:

**Variable-Level Causal Reliability from Heterogeneous Causal Discovery Methods in Multivariate Time Series**

## Contents

### `paper_results_extracted/`
Main publication-run outputs, including synthetic stress tests, family-composition analysis, sensitivity analyses, Lorenz-96, CausalTime, configuration metadata, and publication-gate checks.

### `postprocess_theta050/`
External benchmark post-processing at the optional operating threshold \(\theta_V=0.50\).

### `postprocess_score_diagnostic/`
Threshold-free ranking analysis, including Average Precision, ROC-AUC, paired Wilcoxon tests, pooled unique-target analysis, and Holm-corrected domain-level tests.

### `paper_results_20261010_075342.zip`
Archived copy of the final publication-run outputs.

## Final run status

- Synthetic runs: 90/90 complete
- Family-composition rows: 1350
- Threshold-sensitivity rows: 630
- Family-prior rows: 450
- Lorenz-96 rows: 100
- CausalTime rows: 684

## Main threshold-free external result

Pooled over 77 evaluable external targets:

- Average Precision: 0.4856 vs 0.4635, paired Wilcoxon p = 0.00004
- ROC-AUC: 0.6373 vs 0.6275, paired Wilcoxon p = 0.03335

After Holm correction, the clearest domain-level result is CausalTime-PM2.5 Average Precision, adjusted p = 0.00147.

The archived results are provided so reviewers can inspect the reported evidence without rerunning the full computationally expensive causal-discovery pipeline.
