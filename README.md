# 🚦 CausalGap: Benchmarking Query Correctness for Causal Analysis over Versioned Traffic Data**

CausalGap checks whether traffic-forecast comparisons remain valid when road graphs and cached predictions change. It covers a METR-LA checkpoint replay, paired-record integrity checks, and SQL execution studies in SQLite and DuckDB.

## 💡 Main contributions

- 🔗 **Version-aware comparison keys.** Match predictions using checkpoint, graph version, time, sensor and horizon. The SQL contract detects missing partners, duplicate keys, mismatched graph versions and unequal validity masks.
- 🔍 **Source-to-query checks.** Trace graph changes from STGCN/DSAI inference through caches and saved predictions to SQL summaries. Compare cold and warm executions and inject controlled record faults.
- ⚡ **Correctness and query cost.** Measure checked joins, materialized pair reuse, index choices, resource settings and output granularity. The graph-response experiments examine the H1/H2 analysis using fixed observed targets.

## 🧪 Experiments

| Workload | Code | Scope |
|---|---|---|
| 🔁 Graph cache regression | `cache_replay/` | Compare the original cache keys with byte-accurate graph keys |
| 🚦 METR-LA replay | `real_replay/` | Train or replay three checkpoints with seeds 7, 17 and 27 |
| 🔬 Graph-input analysis | `reconstruction/` | 5%, 10% and 20% graph edits, H1/H2 and source-linked SQL |
| 🗃️ Database correctness | `db_experiments/` | Deterministic forecast records, exact answers, fault injection and query plans |
| 📈 Query resources | `db_extension/` | Randomized SQLite/DuckDB blocks, CPU, memory and thread settings |
| 📊 Grouping granularity | `db_granularity/` | Run, origin, sensor and horizon aggregation |

The real-data experiments hold observed future targets fixed while changing graph input. They measure model-input sensitivity, not physical traffic interventions. The relational workload uses generated integer records. Its 21 model IDs are database keys, **not** 21 newly fitted models.

## 🗂️ Datasets

| Dataset | # Nodes | # Timesteps | Time Span | Interval | Type |
|:---|---:|---:|:---|:---|:---|
| NYCTaxi | 200 | 2,880 | Jan-Mar 2015 | 30 min | Taxi flow |
| NYCBike1 | 128 | 4,392 | Apr-Sep 2014 | 1 hour | Bike flow |
| NYCBike2 | 200 | 2,976 | Jul-Aug 2016 | 30 min | Bike flow |
| BJTaxi | 1024 | 5,596 | Mar-Jun 2015 | 30 min | Taxi flow |
| METR-LA | 207 | 34,272 | Mar-Jun 2012 | 5 min | Traffic speed |
| PeMS-BAY | 325 | 52,116 | Jan-May 2017 | 5 min | Traffic speed |
| PeMS04 | 307 | 16,992 | Jan-Feb 2018 | 5 min | Traffic flow |
| PeMS08 | 170 | 17,856 | Jul-Aug 2016 | 5 min | Traffic flow |

Datasets are available at [Google Drive](https://drive.google.com/file/d/1rlYNzV9O6bBuT6plmYAc5RuFwFFQKZI5/view?usp=sharing).

## 📂 Repository structure

```text
📦 CausalGap/
├── 📘 README.md
├── 📄 requirements.txt
├── 🧠 causalgap/                 # STGCN, DSAI, graph utilities
│   ├── spatial.py
│   └── models/
│       ├── base.py
│       ├── dsai.py
│       └── stgcn.py
├── 🔁 cache_replay/             # Graph cache regression
│   └── run_regression.py
├── 🚦 real_replay/              # Training and forecast replay
│   ├── train_metr_la.py
│   ├── replay_checkpoints.py
│   ├── replay_sqlite.py
│   ├── validate_outputs.py
│   └── measure_keys.py
├── 🔬 reconstruction/           # Graph edits, H1/H2, SQL pairing
│   ├── run_graph_input.py
│   ├── verify_graph_input.py
│   ├── analyze_graph_response.py
│   ├── build_graph_sqlite.py
│   ├── check_graph_pairing.py
│   └── export_prediction_pairs.py
├── 🗃️ db_experiments/           # Query generator and fault tests
│   ├── workload.py
│   ├── queries.py
│   ├── benchmark.py
│   ├── corruption.py
│   └── analyze.py
├── ⚡ db_extension/             # Randomized resource benchmark
│   ├── run_extension.py
│   └── analyze_extension.py
├── 📊 db_granularity/           # Four SQL grouping levels
│   ├── queries.py
│   ├── prepare_inputs.py
│   ├── preflight.py
│   ├── run_granularity.py
│   └── analyze_granularity.py
├── 🛠️ scripts/
│   └── prepare_db_inputs.py   # Generate the SQL test records
├── 🧩 artifact/
│   └── benchmark_contract.sql
└── ✅ tests/
    └── check_contract.py
```

## ⚙️ Installation

Use Python 3.12 or a compatible environment. SQLite is provided by Python, and DuckDB is installed from `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 🚀 Running the code

Run the commands from the repository root.

**1. SQL correctness and two-layer graph cache tests**

```bash
python tests/check_contract.py
python cache_replay/run_regression.py
python db_experiments/corruption.py --output-dir db_experiments/results_faults
```

**2. METR-LA replay and graph-input analysis**

Download `metr-la.h5` first. Training creates the test windows and checkpoint files that the later scripts need. The training, graph edits and H1/H2 calculations follow the settings of the revised study.

```bash
python real_replay/train_metr_la.py
python real_replay/replay_checkpoints.py
python real_replay/replay_sqlite.py
python real_replay/measure_keys.py
python reconstruction/run_graph_input.py
python reconstruction/verify_graph_input.py
python reconstruction/analyze_graph_response.py
python reconstruction/build_graph_sqlite.py
python reconstruction/check_graph_pairing.py
```

**3. SQL performance and output granularity**

First generate the database input. No downloadable SQL benchmark data is required.

```bash
python scripts/prepare_db_inputs.py --scale large
python db_granularity/prepare_inputs.py
python db_granularity/preflight.py
```

The full timing studies are designed for Linux and use CPU affinity and `/proc` information. Their results depend on the host hardware.

```bash
python db_experiments/benchmark.py --output-dir measurements --repeats 5
python db_experiments/analyze.py --measurements measurements
python db_extension/run_extension.py --output measurements
python db_extension/analyze_extension.py
python db_granularity/run_granularity.py --output measurements
python db_granularity/analyze_granularity.py
```

The original runs used 20 randomized blocks for the resource and granularity studies. The scripts generate their own CSV/JSON measurements, SQL plans and plots when executed.

## ✅ Checks and reproducibility

- Graph-input comparisons use checkpoints 7, 17 and 27 with a graph-edit seed of 8431.
- The baseline and changed graph share observed target windows and sensor order.
- H1 holds out an entire checkpoint seed. H2 uses 56 windows for fitting the correction head and 112 later windows for evaluation.
- SQL pair checks compare expected support and complete validity masks before accepting a result.
- The large generated SQL workload contains 1,197,504 forecast atoms and 299,376 validated pairs for the selected interface.
- No generated results or historical measurement files are included, so the published numeric timings cannot be verified from this ZIP alone. They require running the benchmarks in the documented environment.
