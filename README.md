# Variable-Level Causal Reliability

Reproducibility materials for the ACIJ manuscript:

**Variable-Level Causal Reliability from Heterogeneous Causal Discovery Methods in Multivariate Time Series**

Authors:
- Odil Mamaraufov
- Gulrukhora Boykhurozova
- Shakhlo Khabibullayeva
- Akbar Qurbonov

## What this repository contains

This repository provides the final experimental outputs and the code required to inspect and reproduce the evaluation of the family-normalized variable-level causal reliability score.

The experimental study includes:

- 9 synthetic stress scenarios × 10 seeds = 90 datasets;
- 1:1:1, 2:1:1, and 3:1:1 method-family composition experiments;
- controlled family-imbalance patterns;
- threshold sensitivity;
- family-prior sensitivity;
- Lorenz-96 evaluation;
- CausalTime medical, PM2.5, and traffic evaluation;
- threshold-free Average Precision and ROC-AUC;
- paired Wilcoxon signed-rank tests with Holm correction.

## Method families

- Conditional independence: PCMCI+, PCMCI, LPCMCI
- Non-Gaussian SVAR: VAR-LiNGAM
- Continuous-score optimization: DYNOTEARS

## Repository structure

```text
code/
data/
    synthetic/
    lorenz96/
    causaltime/
results/
docs/
README.md
CITATION.cff
```

## Final experiment status

The archived publication gate confirms:

- synthetic runs: 90/90 complete;
- controlled family-imbalance check: passed;
- family-composition rows: 1350;
- threshold-sensitivity rows: 630;
- family-prior rows: 450;
- Lorenz-96 rows: 100;
- CausalTime rows: 684.

## External data

CausalTime is an external public benchmark. It should be obtained from its official upstream distribution. This repository contains our derived evaluation outputs and should contain our preprocessing/evaluation code, but the external benchmark itself is not redistributed here by default.

## Important code file

The exact final Colab notebook used for the publication run must be added as:

`code/CausalTemporalFS_ACIJ_Final.ipynb`

Original notebook name used during the study:

`CausalTemporalFS_v08_ACIJ_FINAL_RUN_CLEAN(1).ipynb`

The current archive contains the final results but **does not yet contain that notebook**. Add the exact notebook before publishing the GitHub repository.

## Main threshold-free external result

Pooled over 77 evaluable external targets:

- AP: 0.4856 vs 0.4635, paired Wilcoxon p = 0.00004
- ROC-AUC: 0.6373 vs 0.6275, p = 0.03335

After Holm correction, the clearest domain-level result is CausalTime-PM2.5 AP, adjusted p = 0.00147.

The method is therefore presented as a reliability/ranking representation, not as a universally superior fixed-threshold binary selector.
