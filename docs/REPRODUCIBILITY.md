# Reproducibility Notes

## Public code

The repository provides two code forms:

- `code/CausalTemporalFS_ACIJ_Reproducible.ipynb`
- `code/causal_temporal_fs_acij.py`

The notebook is intended for Google Colab/Jupyter execution. The Python file is a source-oriented analogue suitable for code review and Git-based comparison.

## Original final experiment archive

The final experiment workflow produced an archived publication run containing:

- 90 synthetic stress runs;
- family-composition experiments;
- threshold sensitivity;
- family-prior sensitivity;
- Lorenz-96 evaluation;
- CausalTime evaluation;
- threshold-free AP/ROC-AUC diagnostics;
- paired Wilcoxon and Holm-corrected statistical results.

## Computational cost

A complete rerun is computationally expensive because multiple causal-discovery algorithms are applied repeatedly across synthetic and external benchmarks.

For this reason, the repository also includes archived final result files under `results/`, allowing the reported tables and statistical analyses to be inspected without repeating every base-method computation.

## Reproduction policy

To reproduce the study:

1. Install the packages in `requirements.txt`.
2. Obtain the external CausalTime benchmark from its official upstream source.
3. Place the required CausalTime files in the path expected by the notebook or adjust the path configuration.
4. Execute the reproducibility notebook from top to bottom.
5. Compare regenerated outputs with the archived files under `results/`.

The published protocol should be preserved: do not tune graph orientation, family weights, thresholds, or method composition after observing benchmark performance.
