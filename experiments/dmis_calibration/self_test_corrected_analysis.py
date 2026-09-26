#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("a", HERE/"analyze_calibrated_diagnostics.py")
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

# Mid-rank CDF formula exact checks.
ref=np.array([.1,.1,.2,.2,.4,.4])
m,lt,eq=a.empirical_magnitude_map(.1,ref)
assert lt==0 and eq==2 and abs(m-(1+1)/(7))<1e-12, (m,lt,eq)
m,lt,eq=a.empirical_magnitude_map(.2,ref)
assert lt==2 and eq==2 and abs(m-(1+2+1)/7)<1e-12
m,lt,eq=a.empirical_magnitude_map(.6,ref)
assert lt==6 and eq==0 and abs(m-1.0)<1e-12

# Cross-family calibration must use DZ / d_tilde, not DZ / raw d_T.
df=pd.DataFrame({
    "dataset":["D","D"],"intervention_type":["node_isolation","node_isolation"],
    "environment_distance":[.1,.6],"representation_response_raw":[.2,.2],
})
cal=pd.DataFrame({"dataset":["D"]*6,"intervention_type":["node_isolation"]*6,"environment_distance":ref})
out=a.apply_cross_family_calibration(df,cal,1e-6)
assert np.allclose(out["DMIS_cal"].to_numpy(), .2/out["environment_distance_calibrated"].to_numpy())
assert not np.allclose(out["DMIS_cal"].to_numpy(), .2/out["environment_distance"].to_numpy())
print("[PASS] corrected Section 5.1 calibration/analysis self-test")
