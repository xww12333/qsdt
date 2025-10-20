Copernicus Pipeline (Minimal Reproducible Skeleton)

Purpose
- Provide a clean, reproducible structure to run QSDT “Copernicus” predictions end-to-end.
- Separate inputs, calibration, RG running, predictions, and validation.

Structure
- config.yaml: All inputs (constants, datasets, scales, toggles) in one place.
- pipeline.py: Orchestrates steps with deterministic logging.
- rg.py: Beta-function interfaces and integrators (placeholders to be filled).
- models.py: Domain mapping from RG outputs to observables (placeholders).
- data/: Holds calibration and holdout datasets (CSV/JSON).
- outputs/: Artifacts (logs, npz, csv) written here.

Workflow
1) Calibrate: Fit/lock unique RG law from calibration set.
2) Run RG: Evolve parameters from µ0 → target scales.
3) Predict: Compute observables with no free parameters.
4) Validate: Compare to holdout set; produce residuals/metrics.

Notes
- No domain formulas are hard-coded here. Plug in functions in rg.py/models.py.
- Deterministic seeds; version the config+code to ensure reproducibility.

Plugin formulas
- You can supply authoritative QSDT mappings via a plugin module.
- Create a module under `scripts/copernicus/plugins/` and set in config:
  `predictions.formulas_module: scripts.copernicus.plugins.strict_formulas`
- Implement `predict_at_mu_override(params, mu, observables, cfg)` to return computed values.
