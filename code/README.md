# Reproducibility Code

This directory contains the publication-facing implementation used to reproduce the experiments reported in the ACIJ manuscript:

**Variable-Level Causal Reliability from Heterogeneous Causal Discovery Methods in Multivariate Time Series**

## Files

### `CausalTemporalFS_ACIJ_Reproducible.ipynb`

Google Colab/Jupyter version of the experimental pipeline.

It contains the scientific workflow used for:

- synthetic stress experiments;
- 1:1:1, 2:1:1, and 3:1:1 method-family composition analysis;
- controlled family-imbalance experiments;
- threshold sensitivity;
- family-prior sensitivity;
- Lorenz-96 evaluation;
- CausalTime medical, PM2.5, and traffic evaluation;
- threshold-free Average Precision and ROC-AUC analysis;
- paired Wilcoxon signed-rank tests and Holm correction.

This publication-facing notebook was derived from the final experimental notebook. Development-only smoke/recovery/display cells, stored cell outputs, and transient notebook metadata were removed while preserving the scientific computation and experiment order.

### `causal_temporal_fs_acij.py`

Source-oriented Python analogue of the reproducibility notebook.

The file uses `# %%` cell separators so that it can be:

- reviewed easily in GitHub;
- compared through Git diffs;
- opened as an interactive script in VS Code, Spyder, or compatible IDEs.

## Main methodological components

The code implements the family-normalized variable-level causal reliability score

\[
R_V(X_i)=\sum_{f\in\mathcal F}\pi_f S_f(X_i),
\]

where method support is normalized within methodological families before aggregation across families.

The method families used in the main experiments are:

- **Conditional independence:** PCMCI+, PCMCI, LPCMCI
- **Non-Gaussian SVAR:** VAR-LiNGAM
- **Continuous-score optimization:** DYNOTEARS

## External benchmark note

CausalTime is an external benchmark and is not redistributed in this repository by default. The code expects the benchmark files to be available locally or in the configured Google Drive location.

## Computational note

A complete rerun of all causal-discovery methods is computationally expensive. The repository therefore also provides archived final outputs under `../results/` so that the reported tables and statistical analyses can be inspected without repeating every base-method run.
