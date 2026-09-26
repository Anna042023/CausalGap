# Abstract SUMO Replication

Runs the simulation-based replication of model response, prediction degradation, cross-setting ranking, and operational-regime analyses.

Start with the two-model pilot before launching the full matrix:

```bash
bash 00_stage_and_check.sh
bash 01_two_model_pilot.sh
bash 02_all_models.sh
bash 03_analyze_replication.sh
```
