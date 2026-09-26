# Incremental Diagnostic Value

Implements the weak-to-strong H1 analysis in controlled synthetic dynamics. The pipeline builds intervention environments, evaluates frozen model checkpoints, constructs prediction units, and runs the nested ridge analysis used to test the incremental value of DMIS beyond weak-error history.

```bash
python run_h1.py --help
```

`run_primary_synthetic.sh` runs the controlled synthetic setting; `audit_h1.py` performs the final consistency audit.
