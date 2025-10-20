Data schema for Copernicus pipeline

Files (optional but recommended):

1) calibration.csv
   - Purpose: values used to lock RG law (fit or set unique parameters)
   - Columns (example): observable, mu_GeV, meas, unc
   - Example rows:
       alpha_em@mu, 91.1876, 0.00729735257, 1.0e-12
       m_e,         0.000511, 0.00051099895, 5.0e-11

2) holdout.csv
   - Purpose: values to validate predictions (not used for calibration)
   - Columns: observable, mu_GeV, meas, unc
   - Example rows:
       Higgs_mass,  125.0,     125.10,      0.14
       delta_m_np,  1.0,       1.293,       0.0001

Notes:
- Observables must match names produced by models.predict_at_mu.
- mu_GeV is the reference RG scale associated with the measurement.

