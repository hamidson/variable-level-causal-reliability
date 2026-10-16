# Data Availability

This study uses internally generated synthetic data, a simulated Lorenz-96 benchmark, and the external CausalTime benchmark.

## Synthetic stress-suite data

The synthetic experiments are generated programmatically from nine predefined scenarios with seeds 101–110. The generation logic and scenario parameters are contained in the reproducibility code.

## Lorenz-96

Lorenz-96 data are generated computationally using the configuration reported in the manuscript and the reproducibility code.

## CausalTime

CausalTime is an external public benchmark. The benchmark itself is not redistributed in this repository by default.

The repository contains the preprocessing/evaluation logic and derived results used in the manuscript. Users should obtain the benchmark from its official upstream distribution and place the required files in the local path expected by the reproducibility code.

The evaluation uses the medical, PM2.5, and traffic domains.
