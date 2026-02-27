# The Tracing Observables using Neural Estimation (T.O.N.E.) Project

## Overview

TONE is a large-scale spectroscopic survey using the Hobby-Eberly Telescope and its VIRUS instrument to detect Lyman-alpha emission from thousands of galaxies.

I lead the end-to-end data analysis pipeline that transforms raw spectroscopic and photometric data into a machine-learning-ready dataset used by Lyra, a Bayesian inference model that predicts emergent Lyman-alpha strength from galaxy properties.

This project involved large-scale catalog cross-matching (>11 million sources), probabilistic modeling, validation, and ML dataset construction.

## Problem Statement

Lyman-alpha emission carries critical physical information about galaxies, but it is difficult to predict due to:

- Noisy spectroscopic detections
- Incomplete photometric coverage
- High-dimensional correlated galaxy properties
- Measurement uncertainty propagation

The goal was to build a reproducible pipeline that:

- Integrates heterogeneous astronomical datasets
- Infers galaxy physical properties
- Quantifies detection confidence
- Produces a structured training set for downstream ML modeling

## Data Scale

- 6 independent photometric survey fields
   - 11 million spectroscopic sources

- Thousands of confirmed galaxy detections

- High-dimensional feature space (SED-derived properties + spectroscopic features)

## Technical Contributions

1. Large-Scale Cross-Matching

- Cross-matched 6 photometric catalogs against a spectroscopic database containing >11M sources
- Implemented efficient spatial joins and duplicate handling
- Resolved ambiguous matches using confidence thresholds
- Built automated validation checks to prevent data leakage

Skills demonstrated: large dataset integration, entity resolution, data cleaning, validation pipelines.

2. Galaxy Property Inference (Feature Engineering)

- Fit spectral energy distributions (SEDs) to derive physical galaxy parameters
- Propagated observational uncertainties into posterior estimates
- Standardized and validated derived features for ML compatibility

Skills demonstrated: probabilistic modeling, feature engineering, uncertainty quantification.

3. Lyman-Alpha Detection Confidence Modeling

- Built criteria to quantify detection reliability
- Implemented threshold-based and probabilistic filtering
- Designed validation framework to assess false positives and false negatives

Skills demonstrated: classification framing, metric selection, error analysis.


4. Validation & Testing Framework

- Performed holdout validation and consistency checks across survey fields
- Verified robustness against selection bias
- Ensured reproducibility across HPC environments

Skills demonstrated: experimental design, bias mitigation, reproducibility.

5. ML-Ready Dataset Construction

- Merged inferred properties and detection labels into structured training data
- Designed schema optimized for downstream modeling (Lyra)
- Delivered clean feature matrices with uncertainty tracking

Skills demonstrated: ML pipeline design, dataset versioning.

## Technologies Used

Python (NumPy, pandas, astropy)
HPC parallelization (Slurm)
Bayesian modeling
Catalog cross-matching algorithms
Large-scale data validation
Git-based reproducibility


## Impact

- Produced a scalable, reproducible dataset enabling probabilistic prediction of Lyman-alpha emission
- Enabled downstream Bayesian ML modeling with calibrated uncertainty
- Built infrastructure capable of handling tens of thousands of galaxies and millions of candidate matches