from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch


EXPERIMENT_NAME = "sumo_incremental_value"
DEFAULT_PROJECT_ROOT = Path("/data/wanganna/CausalGap")
DEFAULT_EXPERIMENT_ROOT = DEFAULT_PROJECT_ROOT / "experiment_results" / EXPERIMENT_NAME

# Actual runnable benchmark set. CityCAN is intentionally absent because the user's
# current codebase has 21 runnable models.
MODEL_FAMILIES = {
    "STGCN": 1,
    "DCRNN": 1,
    "ASTGNN": 1,
    "GWNet": 1,
    "AGCRN": 1,
    "STEVE": 2,
    "DCGCN": 2,
    "ST-Norm": 2,
    "CCHMM": 3,
    "CausalGRIT": 3,
    "TF-MVGNN": 3,
    "PatchTST": 4,
    "iTransformer": 4,
    "TimesNet": 4,
    "FEDformer": 4,
    "DLinear": 4,
    "Autoformer": 4,
    "Chronos": 5,
    "TimesFM": 5,
    "Moirai": 5,
    "Lag-Llama": 5,
}

FAMILY_NAMES = {
    1: "Forecasting STGNNs",
    2: "Robust and Structure-Aware Models",
    3: "Structural-Dynamics Models",
    4: "Generic Time-Series Models",
    5: "Foundation Models",
}

FOUNDATION_MODELS = {m for m, f in MODEL_FAMILIES.items() if f == 5}
ALL_MODELS = list(MODEL_FAMILIES)

# H1 weak -> held-out strong protocol from the revised draft.
# `strength` is always monotone in intervention severity and is used only as
# descriptive metadata / the H1 error-history slope coordinate.
H1_LEVELS = {
    "edge_removal": [
        dict(level="s=0.05", phase="weak", severity=0.05, gamma=None, strength=0.05),
        dict(level="s=0.10", phase="weak", severity=0.10, gamma=None, strength=0.10),
        dict(level="s=0.15", phase="weak", severity=0.15, gamma=None, strength=0.15),
        dict(level="s=0.20", phase="strong", severity=0.20, gamma=None, strength=0.20),
    ],
    "weight_modification": [
        dict(level="gamma=0.8", phase="weak", severity=None, gamma=0.8, strength=0.2),
        dict(level="gamma=0.6", phase="weak", severity=None, gamma=0.6, strength=0.4),
        dict(level="gamma=0.4", phase="weak", severity=None, gamma=0.4, strength=0.6),
        dict(level="gamma=0.2", phase="strong", severity=None, gamma=0.2, strength=0.8),
    ],
    "node_isolation": [
        dict(level="s=0.10", phase="weak", severity=0.10, gamma=None, strength=0.10),
        dict(level="s=0.20", phase="weak", severity=0.20, gamma=None, strength=0.20),
        dict(level="s=0.40", phase="weak", severity=0.40, gamma=None, strength=0.40),
        dict(level="s=0.60", phase="strong", severity=0.60, gamma=None, strength=0.60),
    ],
}

STRUCTURAL_TYPES = list(H1_LEVELS)
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

# Architecture controls C. These flags describe presence of the broad component,
# not family membership. Model/family identity indicators are never used in H1.
ARCH_FLAGS = {
    "STGCN":       dict(recurrent=0, convolutional=1, attention=0),
    "DCRNN":       dict(recurrent=1, convolutional=1, attention=0),
    "ASTGNN":      dict(recurrent=0, convolutional=1, attention=1),
    "GWNet":       dict(recurrent=0, convolutional=1, attention=0),
    "AGCRN":       dict(recurrent=1, convolutional=1, attention=0),
    "STEVE":       dict(recurrent=0, convolutional=0, attention=1),
    "DCGCN":       dict(recurrent=0, convolutional=1, attention=0),
    "ST-Norm":     dict(recurrent=0, convolutional=1, attention=0),
    "CCHMM":       dict(recurrent=1, convolutional=0, attention=0),
    "CausalGRIT":  dict(recurrent=1, convolutional=0, attention=1),
    "TF-MVGNN":    dict(recurrent=1, convolutional=1, attention=0),
    "PatchTST":    dict(recurrent=0, convolutional=0, attention=1),
    "iTransformer":dict(recurrent=0, convolutional=0, attention=1),
    "TimesNet":    dict(recurrent=0, convolutional=1, attention=0),
    "FEDformer":   dict(recurrent=0, convolutional=0, attention=1),
    "DLinear":     dict(recurrent=0, convolutional=0, attention=0),
    "Autoformer":  dict(recurrent=0, convolutional=0, attention=1),
    "Chronos":     dict(recurrent=0, convolutional=0, attention=1),
    "TimesFM":     dict(recurrent=0, convolutional=0, attention=1),
    "Moirai":      dict(recurrent=0, convolutional=0, attention=1),
    "Lag-Llama":   dict(recurrent=0, convolutional=0, attention=1),
}


def family_of(model: str) -> int:
    if model not in MODEL_FAMILIES:
        raise KeyError(f"Unknown H1 model {model!r}")
    return MODEL_FAMILIES[model]


def family_name(model_or_family) -> str:
    f = family_of(model_or_family) if isinstance(model_or_family, str) else int(model_or_family)
    return FAMILY_NAMES[f]


def normalized_weak_strength(kind: str, strength: float) -> float:
    vals = [float(x["strength"]) for x in H1_LEVELS[kind] if x["phase"] == "weak"]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return 0.0
    return float((float(strength) - lo) / (hi - lo))


def plan_rows(intervention_seeds: Iterable[int]):
    rows = []
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            for seed in intervention_seeds:
                r = dict(spec)
                r.update(intervention_type=kind, intervention_seed=int(seed))
                r["normalized_weak_intensity"] = (
                    normalized_weak_strength(kind, r["strength"]) if r["phase"] == "weak" else np.nan
                )
                rows.append(r)
    return rows


def plan_frame(intervention_seeds: Iterable[int]) -> pd.DataFrame:
    return pd.DataFrame(plan_rows(intervention_seeds))


def intensity_lookup(kind: str, level: str) -> dict:
    for spec in H1_LEVELS[kind]:
        if spec["level"] == level:
            return dict(spec)
    raise KeyError((kind, level))


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def json_dump(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_state_dict_like(obj) -> int | None:
    """Recursively find a tensor state dict in common PyTorch checkpoint layouts."""
    if isinstance(obj, dict):
        direct = [v for v in obj.values() if torch.is_tensor(v)]
        if direct and len(direct) >= max(1, len(obj) // 3):
            return int(sum(int(v.numel()) for v in direct))
        for key in ("state_dict", "model", "module", "model_state_dict", "params"):
            if key in obj:
                got = _count_state_dict_like(obj[key])
                if got is not None:
                    return got
        for v in obj.values():
            got = _count_state_dict_like(v)
            if got is not None:
                return got
    return None


def count_checkpoint_parameters(path: Path) -> tuple[int | None, str]:
    """Count parameters from the exact local checkpoint when possible.

    This avoids hard-coding published parameter counts for potentially different
    local checkpoint variants. It is used for the H1 log-parameter-count control.
    """
    path = Path(path)
    try:
        from safetensors import safe_open
    except Exception:
        safe_open = None

    candidates = []
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        # Prefer an index if sharded; otherwise inspect safetensors/checkpoints.
        candidates = sorted(path.rglob("*.safetensors"))
        if not candidates:
            candidates = (sorted(path.rglob("*.ckpt")) + sorted(path.rglob("*.pt")) +
                          sorted(path.rglob("*.pth")) + sorted(path.rglob("*.bin")))

    if not candidates:
        return None, "no_checkpoint_tensor_file_found"

    total = 0
    used = 0
    # Safetensors shards can be summed safely. For generic torch files use the
    # first checkpoint that yields a state dict to avoid double-counting copies.
    st_files = [p for p in candidates if p.suffix == ".safetensors"]
    if st_files and safe_open is not None:
        try:
            for p in st_files:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        total += int(np.prod(f.get_slice(k).get_shape()))
                used += 1
            if used:
                return int(total), f"safetensors:{used}_file(s)"
        except Exception:
            pass

    for p in candidates:
        if p.suffix == ".safetensors":
            continue
        try:
            try:
                obj = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(p, map_location="cpu")
            got = _count_state_dict_like(obj)
            if got is not None:
                return int(got), f"torch_checkpoint:{p.name}"
        except Exception:
            continue
    return None, "checkpoint_parameter_count_failed"



def foundation_weight_path(model: str, project_root: Path) -> Path | None:
    """Return the benchmark's configured local foundation-weight location."""
    project_root = Path(project_root)
    if model == "Chronos":
        return project_root / "model_weights" / "Chronos"
    if model == "TimesFM":
        return project_root / "model_weights" / "TimesFM" / "torch_model.ckpt"
    if model == "Moirai":
        return project_root / "model_weights" / "Moirai"
    if model == "Lag-Llama":
        return project_root / "model_weights" / "Lag-Llama" / "lag-llama.ckpt"
    return None


def resolve_foundation_parameter_count(model: str, project_root: Path,
                                       explicit_counts: dict[str, int] | None = None) -> tuple[int | None, str]:
    """Resolve the frozen foundation backbone's real parameter count.

    V1 could silently fall back to a tiny local wrapper (e.g. 12 parameters) when
    the external checkpoint could not be counted. V2 forbids that fallback for H1
    controls. An explicit audited override may be supplied, otherwise the exact
    local checkpoint is inspected.
    """
    if explicit_counts and model in explicit_counts:
        v = int(explicit_counts[model])
        if v <= 0:
            return None, "invalid_explicit_override"
        return v, "explicit_audited_override"
    path = foundation_weight_path(model, project_root)
    if path is None:
        return None, "no_foundation_path"
    got, src = count_checkpoint_parameters(path)
    if got is None:
        return None, f"unresolved:{src}:{path}"
    # Guard against accidentally counting only a wrapper/config tensor collection.
    if int(got) < 100_000:
        return None, f"implausibly_small_external_count:{got}:{src}:{path}"
    return int(got), f"checkpoint:{src}:{path}"

def model_control_record(model_name: str, constructed_model=None, native_uses_graph: bool | None = None,
                         foundation_weight_path: Path | None = None) -> dict:
    fam = family_of(model_name)
    flags = ARCH_FLAGS[model_name]
    local_count = None
    if constructed_model is not None:
        local_count = int(sum(int(p.numel()) for p in constructed_model.parameters()))

    external_count = None
    source = "constructed_trackB_model"
    if fam == 5 and foundation_weight_path is not None:
        external_count, external_source = count_checkpoint_parameters(foundation_weight_path)
        source = external_source

    if fam == 5:
        # The DSAI wrapper is resident in the main process while the frozen
        # foundation backbone lives in the worker process. Add both when the
        # external checkpoint count is available.
        if external_count is not None:
            parameter_count = int(external_count + (local_count or 0))
            source = source + "+local_wrapper"
        else:
            parameter_count = int(local_count or 0)
            source = "local_wrapper_only_missing_external_count"
    else:
        parameter_count = int(local_count or 0)

    return {
        "model": model_name,
        "model_family": fam,
        "model_family_name": FAMILY_NAMES[fam],
        "parameter_count": parameter_count,
        "log_parameter_count": float(math.log1p(max(0, parameter_count))),
        "parameter_count_source": source,
        "native_topology_access": int(bool(native_uses_graph)) if native_uses_graph is not None else np.nan,
        "recurrent_flag": int(flags["recurrent"]),
        "convolutional_flag": int(flags["convolutional"]),
        "attention_flag": int(flags["attention"]),
        "pretraining_flag": int(fam == 5),
        # This matches the user's current code. The revised manuscript currently
        # mentions four Foundation adaptation levels; those are not fabricated here.
        "adaptation_level": "zero_shot" if fam == 5 else "dataset_trained",
    }
