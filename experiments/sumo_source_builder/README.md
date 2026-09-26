# SUMO Source Builder

Builds the road network, maps sensors, prepares demand targets, calibrates demand, generates intervention environments, and validates the resulting SUMO source used by the replication experiments.

Typical order:

```bash
python 01_build_osm_network.py --help
python 02_map_sensors.py --help
python 03_prepare_pems_flow_template.py --help
python 04_calibrate_demand_spsa_paper.py --help
python 05_generate_h1_source.py --help
python 06_validate_source.py --help
```
