# Variable-Level Causal Reliability

Reproducibility materials for the ACIJ manuscript:

**Variable-Level Causal Reliability from Heterogeneous Causal Discovery Methods in Multivariate Time Series**

Authors:
- Odil Mamaraufov
- Gulrukhora Boykhurozova
- Shakhlo Khabibullayeva
- Akbar Qurbonov

## Overview

This repository contains the publication-facing code, archived experimental outputs, and supporting documentation for the family-normalized variable-level causal reliability framework.

The study evaluates:
- 9 synthetic stress scenarios × 10 seeds = 90 datasets;
- 1:1:1, 2:1:1, and 3:1:1 method-family compositions;
- controlled family-imbalance patterns;
- threshold sensitivity;
- family-prior sensitivity;
- Lorenz-96;
- CausalTime medical, PM2.5, and traffic domains;
- binary Precision/Recall/F1;
- threshold-free Average Precision and ROC-AUC;
- paired Wilcoxon signed-rank tests with Holm correction.

## Method families

- **Conditional independence:** PCMCI+, PCMCI, LPCMCI
- **Non-Gaussian SVAR:** VAR-LiNGAM
- **Continuous-score optimization:** DYNOTEARS

## Repository structure

```text
code/
    CausalTemporalFS_ACIJ_Reproducible.ipynb
    causal_temporal_fs_acij.py
    README.md
docs/
    DATA_AVAILABILITY.md
    RESULTS_CHECK.md
    REPRODUCIBILITY.md
    UPLOAD_CHECKLIST.md
results/
    archived experiment outputs and post-processing results
README.md
CITATION.cff
requirements.txt
```

## Code

`code/CausalTemporalFS_ACIJ_Reproducible.ipynb` is the publication-clean reproducibility notebook derived from the final experimental notebook used for the study. Scientific computation and experiment order are preserved, while development-only smoke/recovery/display cells, execution outputs, and transient notebook metadata were removed.

`code/causal_temporal_fs_acij.py` is a source-oriented analogue of the same experimental pipeline. It uses `# %%` cell separators so it can be reviewed easily on GitHub and run interactively in compatible IDEs.

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

Synthetic and Lorenz-96 data are generated computationally by the code.

CausalTime is an external public benchmark and is not redistributed here by default. The repository contains our preprocessing/evaluation logic and derived results. Users should obtain CausalTime from its official upstream distribution.

## Main threshold-free external result

Pooled over 77 evaluable external targets:
- AP: 0.4856 vs 0.4635, paired Wilcoxon p = 0.00004
- ROC-AUC: 0.6373 vs 0.6275, p = 0.03335

After Holm correction, the strongest domain-level result is CausalTime-PM2.5 AP, adjusted p = 0.00147.

These results support family normalization as a continuous reliability/ranking representation rather than as a universally superior fixed-threshold binary selector.

## Reproducibility note

The full causal-discovery experiment is computationally expensive. Archived final outputs are provided so that reported tables and statistical analyses can be inspected without rerunning all base-method computations.
