# CausalGap: Benchmarking Representation Responses in Traffic Prediction under Structural Interventions

**CausalGap** is a diagnostic framework for studying how spatio-temporal forecasting models respond to controlled structural changes. Instead of evaluating forecasting error alone, the framework jointly measures **representation response** and **prediction degradation** under graph interventions, then uses the resulting diagnostic landscape to compare architectures, access protocols, and robustness properties.

The repository contains the benchmark implementation, unified model adapters, intervention generators, foundation-model bridges, and the experiment pipelines used for the paper.

## 🔍 Method overview

For a baseline representation $Z_0$ and an intervened representation $Z_k$, CausalGap first computes a representation-response distance

$$
D_Z = \alpha\,D_{\mathrm{angle}} + (1-\alpha)\,D_{\mathrm{magnitude}},
$$

where $D_{\mathrm{angle}}$ is the centered Frobenius cosine distance and $D_{\mathrm{magnitude}}$ is the centered relative Frobenius change. The **DMIS** diagnostic normalizes this response by the strength of the structural intervention:

$$
\mathrm{DMIS}_k = \frac{D_{Z,k}}{\max(d_{T,k},\epsilon)}.
$$

Prediction sensitivity is summarized by **RD**:

$$
\mathrm{RD}_k = \frac{\mathrm{IPE}_k-\mathrm{IPE}_0}{\mathrm{IPE}_0 + 0.05\,\mathrm{std}(Y_0)}.
$$

Together, DMIS and RD define four operational regimes:

- **Stable Invariance** — low representation response, low prediction degradation
- **Adaptive Response** — high representation response, low prediction degradation
- **Over-Reactive Instability** — high representation response, high prediction degradation
- **Pathological Rigidity** — low representation response, high prediction degradation

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

## 🧠 Model coverage

The benchmark contains 21 runnable architectures grouped into five families:

- **Family I — Forecasting STGNNs:** STGCN, DCRNN, ASTGNN, GWNet, AGCRN
- **Family II — Robust / structure-aware models:** STEVE, DCGCN, ST-Norm
- **Family III — Structural-dynamics models:** CCHMM, CausalGRIT, TF-MVGNN
- **Family IV — Generic time-series models:** PatchTST, iTransformer, TimesNet, FEDformer, DLinear, Autoformer
- **Family V — Foundation models:** Chronos, TimesFM, Moirai, Lag-Llama

Official source trees / local checkpoints are expected under the project asset root configured by `CAUSALGAP_ROOT`.

## 📦 Repository structure

```text
📦 CausalGap/
├── 🧠 causalgap/                       # Core benchmark library
│   ├── models/                        # Unified model adapters and registry
│   ├── data.py                        # Dataset loading and chronological splits
│   ├── interventions.py               # Structural / temporal interventions
│   ├── evaluator.py                   # DMIS, RD and intervention evaluation
│   ├── metrics.py                     # Prediction and representation metrics
│   └── trainer.py                     # Shared supervised training utilities
├── ⚙️ configs/                         # Model / adaptation configuration files
├── 🧪 experiments/
│   ├── observational_diagnostics/     # Main DMIS–RD benchmark on observed datasets
│   ├── dmis_calibration/              # Calibrated diagnostic analysis
│   ├── diagnostic_robustness/         # Multi-seed robustness checks
│   ├── diagnostic_visualization/      # Paper-ready DMIS–RD figures
│   ├── incremental_value/             # H1 weak-to-strong diagnostic value
│   ├── sumo_source_builder/           # SUMO network / demand preparation
│   ├── sumo_incremental_value/        # H1 analysis on SUMO-derived environments
│   ├── sumo_replication/              # Abstract SUMO replication and regime analysis
│   ├── regime_stability/              # Quantile / clustering stability of regimes
│   ├── matched_error/                 # Prediction-error-matched diagnostic analysis
│   ├── representation_robustness/     # CKA / SVCCA / RSA / PWCCA checks
│   ├── invariance_checks/             # Rotation / permutation / scaling invariance
│   ├── representation_steering/       # Information-matched steering experiments
│   ├── access_sensitivity/            # Track A vs. Track B / topology-input stress
│   └── alpha_sensitivity/             # DMIS mixing-weight sensitivity
├── 🤖 foundation_workers/              # Chronos / TimesFM / Moirai / Lag-Llama bridges
├── 🛠️ tools/                           # Environment packing and utility scripts
├── 🚀 scripts/                         # Smoke tests and protocol examples
├── 📄 requirements.txt
└── 📘 README.md
```

## ▶️ Main experiment entry points

Run commands from the repository root. Most scripts also expose `--help` for paths, devices, seeds, and output locations.

| Experiment | Entry point |
|---|---|
| Observational DMIS–RD benchmark | `python experiments/observational_diagnostics/run_diagnostics.py --help` |
| Calibrated diagnostic analysis | `bash experiments/dmis_calibration/run_corrected_analysis.sh` |
| Incremental diagnostic value (H1) | `python experiments/incremental_value/run_h1.py --help` |
| SUMO source construction | `python experiments/sumo_source_builder/01_build_osm_network.py --help` |
| Abstract SUMO replication | `bash experiments/sumo_replication/01_two_model_pilot.sh` |
| Regime stability | `python experiments/regime_stability/run_regime_stability.py --help` |
| Matched-error analysis | `python experiments/matched_error/run_matched_error.py --help` |
| Alternative representation metrics | `python experiments/representation_robustness/analyze_representation_metrics.py --help` |
| Invariance checks | `python experiments/invariance_checks/run_invariance_checks.py --help` |
| Representation steering | `python experiments/representation_steering/run_steering.py --help` |
| Structural-access sensitivity | `python experiments/access_sensitivity/analyze_access_sensitivity.py --help` |
| Mixing-weight sensitivity | `python experiments/alpha_sensitivity/run_alpha_sensitivity.py --help` |

## 🧩 Foundation models

Foundation models are isolated behind subprocess workers because their official implementations have different dependency stacks. The release contains workers for:

- Chronos
- TimesFM
- Moirai
- Lag-Llama

Install each official implementation in its own environment (see `scripts/env/`) and place the corresponding local checkpoints under `$CAUSALGAP_ROOT/model_weights/`. The main `requirements.txt` intentionally contains only the shared benchmark dependencies.

## 📊 Outputs

Experiment scripts write generated artifacts under the selected experiment root, typically with the following structure:

```text
experiment_results/<experiment_name>/
├── runs/             # Per-model checkpoints and intervention evaluations
├── analysis/         # Aggregated analyses
├── tables/           # CSV / LaTeX tables
├── figures/          # PDF / SVG / PNG figures
├── diagnostics/      # Coverage and protocol audits
└── metadata/         # Reproducibility information
```

## ✅ Reproducibility notes

- Model seeds and intervention seeds are kept separate.
- Track A and Track B are evaluated separately and are not pooled into a single ranking.
- Internal structural interventions preserve aligned target support; observational topology-input stress tests are reported separately from intervention-consistent causal claims.
- The representation interface uses row-aligned centered representations before DMIS calculation.
- The default DMIS mixing weight is `alpha = 0.5`; sensitivity scripts recompute alternative values from saved response components without retraining forecasting models.
- Completed interventions are written incrementally where supported so interrupted runs can be resumed.
