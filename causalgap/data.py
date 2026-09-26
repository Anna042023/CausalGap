from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DATA_ROOT, canonical_dataset, dataset_spec
from .graph import load_graph
from .spatial import load_coordinates


_SPEED_DATASETS = {'METR-LA', 'PEMS-BAY'}
_MOBILITY_DATASETS = {'NYCTaxi', 'NYCBike1', 'NYCBike2', 'BJTaxi'}
_FLOW_SENSOR_DATASETS = {'PEMS04', 'PEMS08'}


@dataclass
class StandardScaler:
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def raw_value_to_normalized(self, value: float):
        return (float(value) - self.mean) / self.std


@dataclass
class TrafficDataBundle:
    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_y_mask: np.ndarray
    val_y_mask: np.ndarray
    test_y_mask: np.ndarray
    adj: np.ndarray
    scaler: StandardScaler
    num_nodes: int
    history_len: int
    pred_len: int
    interval_minutes: int
    temporal_protocol: str
    data_source: str
    report_primary_mape: bool
    input_dim: int = 1
    target_dim: int = 1
    split_protocol: str = 'chronological_7_1_2'
    scaler_fit: str = 'train_x_only'
    coords: np.ndarray | None = None
    coordinate_source: str = 'missing_zero_filled'
    coordinate_coverage: float = 0.0
    missing_value_policy: str = 'finite_only'
    speed_missing_policy: str = 'masked_zero'
    speed_missing_input_fill: str = 'keep_zero'
    mape_min_abs: float = 1.0
    train_fully_invalid_samples: int = 0
    val_fully_invalid_samples: int = 0
    test_fully_invalid_samples: int = 0

    @property
    def metric_protocol(self) -> str:
        return 'v3.3_observed_mask_primary_mape_thresholded'

    def target_mask_for_split(self, split: str) -> np.ndarray:
        key = str(split).lower()
        if key == 'train':
            return self.train_y_mask
        if key == 'val':
            return self.val_y_mask
        if key == 'test':
            return self.test_y_mask
        raise KeyError(split)

    def infer_external_target_mask(self, y_original_units: np.ndarray) -> np.ndarray:
        """Build an evaluation mask for a physically generated target array.

        Internal CSG interventions always inherit the exact baseline mask. External
        physical trajectories do not have that baseline mask, so only finite values
        are guaranteed. For speed data using the DCRNN zero-sentinel protocol, raw
        zeros are also treated as unavailable measurements.
        """
        y = np.asarray(y_original_units)
        mask = np.isfinite(y)
        if self.name in _SPEED_DATASETS and self.speed_missing_policy == 'masked_zero':
            mask &= np.abs(y) > 1e-8
        return mask.astype(bool)


class TrafficTensorDataset(Dataset):
    def __init__(self, x, y, y_mask=None):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if y_mask is None:
            y_mask = np.ones_like(y, dtype=bool)
        self.y_mask = torch.from_numpy(np.asarray(y_mask, dtype=bool))
        if self.x.shape[0] != self.y.shape[0] or self.y.shape != self.y_mask.shape:
            raise ValueError(
                f'TrafficTensorDataset shape mismatch: x={self.x.shape}, y={self.y.shape}, mask={self.y_mask.shape}'
            )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_mask[idx]


def _key(z, choices):
    lower = {k.lower(): k for k in z.files}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_xy_npz(path: Path, feature_index=0):
    z = np.load(path, allow_pickle=True)
    kx = _key(z, ['x', 'X'])
    ky = _key(z, ['y', 'Y'])
    if kx is not None and ky is not None:
        x, y = z[kx], z[ky]
    else:
        arrays = [z[k] for k in z.files if isinstance(z[k], np.ndarray)]
        four_d = [a for a in arrays if a.ndim == 4]
        if len(four_d) >= 2:
            x, y = four_d[:2]
        elif len(four_d) == 1:
            raise ValueError(
                f'{path} contains one 4D array but no x/y keys. Cannot safely infer split. keys={z.files}'
            )
        else:
            raise ValueError(f'Cannot find x/y arrays in {path}. keys={z.files}')
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f'Expected 4D x/y in {path}, got {x.shape} and {y.shape}')
    fi_x = min(feature_index, x.shape[-1] - 1)
    fi_y = min(feature_index, y.shape[-1] - 1)
    x = x[..., fi_x:fi_x + 1].astype(np.float32)
    y = y[..., fi_y:fi_y + 1].astype(np.float32)
    return x, y


def _find_raw_pems_npz(dataset_dir: Path, name: str):
    candidates = [dataset_dir / f'{name.lower()}.npz', dataset_dir / f'{name}.npz']
    candidates += list(dataset_dir.glob('*.npz'))
    for p in candidates:
        if not p.exists() or p.name.lower() in ('train.npz', 'val.npz', 'test.npz', 'adj_mx.npz'):
            continue
        try:
            z = np.load(p, allow_pickle=True)
            kd = _key(z, ['data'])
            if kd is not None and z[kd].ndim >= 2:
                return p
        except Exception:
            pass
    return None


def _window_series(arr, history_len, pred_len):
    total = history_len + pred_len
    n = len(arr) - total + 1
    if n <= 0:
        raise ValueError(f'Series too short for history={history_len}, pred={pred_len}: T={len(arr)}')
    xs = np.stack([arr[s:s + history_len] for s in range(n)])
    ys = np.stack([arr[s + history_len:s + total] for s in range(n)])
    return xs, ys


def _split_samples_7_1_2(*arrays):
    """Chronological split AFTER supervised windows are created.

    The same boundaries are applied to x/y/masks so masks can never drift out of
    alignment with their targets.
    """
    if not arrays:
        raise ValueError('No arrays to split.')
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError('All arrays must have the same sample count.')
    b1 = int(n * 0.7)
    b2 = int(n * 0.8)
    if b1 <= 0 or b2 <= b1 or b2 >= n:
        raise ValueError(f'Invalid 7:1:2 split for n={n}')
    out = []
    for a in arrays:
        out.extend([a[:b1], a[b1:b2], a[b2:]])
    return tuple(out)


def _load_raw_pems(dataset_dir: Path, name: str, history_len=12, pred_len=12, feature_index=0):
    p = _find_raw_pems_npz(dataset_dir, name)
    if p is None:
        raise FileNotFoundError(f'Raw {name} npz not found under {dataset_dir}')
    z = np.load(p, allow_pickle=True)
    kd = _key(z, ['data'])
    data = z[kd]
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f'Expected raw data [T,N,F] in {p}, got {data.shape}')
    fi = min(feature_index, data.shape[-1] - 1)
    raw = data[..., fi:fi + 1].astype(np.float32)

    # Preserve which values were actually observed BEFORE zero filling. Genuine
    # observed zero-flow values remain valid for MAE/RMSE; only NaN/Inf gaps are
    # marked invalid. This fixes the old "all zero == missing" ambiguity for flow.
    observed = np.isfinite(raw)
    filled = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    all_x, all_y = _window_series(filled, history_len, pred_len)
    _, all_y_mask = _window_series(observed.astype(np.bool_), history_len, pred_len)
    split = _split_samples_7_1_2(all_x, all_y, all_y_mask)
    train_x, val_x, test_x, train_y, val_y, test_y, train_m, val_m, test_m = split
    return train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m


def _assert_finite(name, *arrays):
    for i, a in enumerate(arrays):
        if not np.isfinite(a).all():
            bad = int(np.size(a) - np.isfinite(a).sum())
            raise ValueError(f'{name}: array #{i} contains {bad} non-finite values after preprocessing.')


def _mask_for_pre_generated(name: str, y: np.ndarray, speed_missing_policy: str) -> np.ndarray:
    mask = np.isfinite(y)
    if name in _SPEED_DATASETS:
        if speed_missing_policy == 'masked_zero':
            mask &= np.abs(y) > 1e-8
        elif speed_missing_policy == 'raw':
            pass
        elif speed_missing_policy == 'interpolate':
            raise RuntimeError(
                f'{name}: speed_missing_policy=interpolate cannot be applied safely to already-windowed '
                'train/val/test NPZ files because chronological raw-series interpolation would have to be '
                'performed before window generation. Use masked_zero with the DCRNN-format files, or '
                'supply/rebuild a raw chronological speed series first.'
            )
        else:
            raise ValueError(f'Unknown speed_missing_policy={speed_missing_policy}')
    return mask.astype(bool)


def _maybe_fill_speed_input(name: str, x: np.ndarray, policy: str, fill: str, train_mean: float | None = None):
    if name not in _SPEED_DATASETS or policy != 'masked_zero' or fill == 'keep_zero':
        return x
    if fill != 'mean':
        raise ValueError(f'Unknown speed_missing_input_fill={fill}')
    out = np.asarray(x, dtype=np.float32).copy()
    invalid = np.abs(out) <= 1e-8
    if train_mean is None:
        valid = np.isfinite(out) & (~invalid)
        train_mean = float(np.mean(out[valid])) if np.any(valid) else 0.0
    out[invalid] = float(train_mean)
    return out


def load_dataset(
    name: str,
    data_root=DATA_ROOT,
    history_len=12,
    pred_len=12,
    feature_index=0,
    speed_missing_policy: str | None = None,
    speed_missing_input_fill: str | None = None,
):
    name = canonical_dataset(name)
    spec = dataset_spec(name)
    speed_missing_policy = (
        speed_missing_policy
        or os.environ.get('CAUSALGAP_SPEED_MISSING_POLICY', 'masked_zero')
    ).lower()
    speed_missing_input_fill = (
        speed_missing_input_fill
        or os.environ.get('CAUSALGAP_SPEED_MISSING_INPUT_FILL', 'keep_zero')
    ).lower()

    dataset_dir = Path(data_root) / name
    if not dataset_dir.exists():
        candidates = [
            p for p in Path(data_root).iterdir()
            if p.name.lower().replace('_', '-') == name.lower().replace('_', '-')
        ]
        if candidates:
            dataset_dir = candidates[0]
        else:
            raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    split_files = [dataset_dir / s for s in ('train.npz', 'val.npz', 'test.npz')]
    if all(p.exists() for p in split_files):
        train_x, train_y = _load_xy_npz(split_files[0], feature_index)
        val_x, val_y = _load_xy_npz(split_files[1], feature_index)
        test_x, test_y = _load_xy_npz(split_files[2], feature_index)
        train_m = _mask_for_pre_generated(name, train_y, speed_missing_policy)
        val_m = _mask_for_pre_generated(name, val_y, speed_missing_policy)
        test_m = _mask_for_pre_generated(name, test_y, speed_missing_policy)
        split_protocol = 'source_pre_generated_chronological_splits'
    elif name in _FLOW_SENSOR_DATASETS:
        (
            train_x, train_y, train_m,
            val_x, val_y, val_m,
            test_x, test_y, test_m,
        ) = _load_raw_pems(dataset_dir, name, history_len, pred_len, feature_index)
        split_protocol = 'window_first_then_chronological_7_1_2_by_sample'
    else:
        raise FileNotFoundError(
            f'{name}: expected train.npz/val.npz/test.npz, or raw PEMS04/08 npz.'
        )

    _assert_finite(name, train_x, train_y, val_x, val_y, test_x, test_y)

    # Optional speed-input mean fill is deliberately separated from the target
    # mask. Default keep_zero preserves existing v3.2 checkpoints and allows a
    # metric-only repair. Switching to mean changes the model inputs and therefore
    # requires retraining; the choice is recorded in run metadata/checkpoints.
    observed_train_x = np.isfinite(train_x)
    if name in _SPEED_DATASETS and speed_missing_policy == 'masked_zero':
        observed_train_x &= np.abs(train_x) > 1e-8
    train_mean_observed = float(np.mean(train_x[observed_train_x])) if np.any(observed_train_x) else float(np.mean(train_x))
    train_x = _maybe_fill_speed_input(name, train_x, speed_missing_policy, speed_missing_input_fill, train_mean_observed)
    val_x = _maybe_fill_speed_input(name, val_x, speed_missing_policy, speed_missing_input_fill, train_mean_observed)
    test_x = _maybe_fill_speed_input(name, test_x, speed_missing_policy, speed_missing_input_fill, train_mean_observed)

    # Keep the historical v3.2 scaler fit (all training inputs) by default so old
    # predictions can be re-scored without changing the scale. Missing-target
    # validity is handled explicitly by masks, not by abusing the scaler.
    mean = float(np.mean(train_x))
    std = float(np.std(train_x))
    if std < 1e-8:
        std = 1.0
    scaler = StandardScaler(mean, std)

    train_x = scaler.transform(train_x).astype(np.float32)
    train_y = scaler.transform(train_y).astype(np.float32)
    val_x = scaler.transform(val_x).astype(np.float32)
    val_y = scaler.transform(val_y).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)
    test_y = scaler.transform(test_y).astype(np.float32)

    num_nodes = int(train_x.shape[2])
    adj = load_graph(dataset_dir, num_nodes)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    coords, coordinate_source, coordinate_mask = load_coordinates(dataset_dir, num_nodes)
    coordinate_coverage = float(np.mean(coordinate_mask.astype(np.float32))) if len(coordinate_mask) else 0.0

    if name in _SPEED_DATASETS:
        missing_value_policy = f'dcrnn_speed_{speed_missing_policy}'
    elif name in _FLOW_SENSOR_DATASETS:
        missing_value_policy = 'flow_preserve_observed_zero_mask_only_nonfinite_gaps'
    else:
        missing_value_policy = 'mobility_finite_only_real_zeros_retained'

    return TrafficDataBundle(
        name=name,
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        train_y_mask=np.asarray(train_m, dtype=bool),
        val_y_mask=np.asarray(val_m, dtype=bool),
        test_y_mask=np.asarray(test_m, dtype=bool),
        adj=adj, scaler=scaler,
        num_nodes=num_nodes,
        history_len=int(train_x.shape[1]),
        pred_len=int(train_y.shape[1]),
        interval_minutes=spec.interval_minutes,
        temporal_protocol=spec.temporal_protocol,
        data_source=spec.source,
        report_primary_mape=spec.report_primary_mape,
        split_protocol=split_protocol,
        scaler_fit='train_x_only',
        coords=coords,
        coordinate_source=coordinate_source,
        coordinate_coverage=coordinate_coverage,
        missing_value_policy=missing_value_policy,
        speed_missing_policy=speed_missing_policy,
        speed_missing_input_fill=speed_missing_input_fill,
        # 1 physical unit avoids exploding ratios from numerical inverse-transform
        # residuals and near-zero denominators. Mobility datasets do not report
        # primary MAPE at all.
        mape_min_abs=1.0,
        train_fully_invalid_samples=int(np.sum(~np.asarray(train_m, dtype=bool).reshape(len(train_m), -1).any(axis=1))),
        val_fully_invalid_samples=int(np.sum(~np.asarray(val_m, dtype=bool).reshape(len(val_m), -1).any(axis=1))),
        test_fully_invalid_samples=int(np.sum(~np.asarray(test_m, dtype=bool).reshape(len(test_m), -1).any(axis=1))),
    )


def _drop_fully_invalid_supervised_samples(x, y, y_mask, split_name: str, dataset_name: str = 'dataset'):
    """Remove rows with zero valid target elements from train/validation.

    Such samples have exactly zero supervised information under the declared
    target-validity policy. Keeping them can create an all-invalid shuffled batch
    and abort training even though the dataset is otherwise healthy. Test rows are
    intentionally preserved so reporting still spans the complete test timeline;
    element-wise masks handle them during evaluation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.asarray(y_mask, dtype=bool)
    if len(x) != len(y) or y.shape != m.shape:
        raise ValueError(f'{split_name}: x/y/mask shape mismatch: {x.shape}, {y.shape}, {m.shape}')
    keep = m.reshape(len(m), -1).any(axis=1) & np.isfinite(y).reshape(len(y), -1).any(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print(
            f'[DATA] {dataset_name}: dropping {dropped}/{len(keep)} fully-invalid '
            f'{split_name} samples (zero supervised weight under target mask).',
            flush=True,
        )
    return x[keep], y[keep], m[keep], dropped


def make_loaders(bundle: TrafficDataBundle, batch_size=32, num_workers=0):
    train_x, train_y, train_m, dropped_train = _drop_fully_invalid_supervised_samples(
        bundle.train_x, bundle.train_y, bundle.train_y_mask, 'train', bundle.name
    )
    val_x, val_y, val_m, dropped_val = _drop_fully_invalid_supervised_samples(
        bundle.val_x, bundle.val_y, bundle.val_y_mask, 'val', bundle.name
    )
    bundle.train_fully_invalid_samples = int(dropped_train)
    bundle.val_fully_invalid_samples = int(dropped_val)
    # Test is deliberately not filtered.
    bundle.test_fully_invalid_samples = int(
        np.sum(~np.asarray(bundle.test_y_mask, dtype=bool).reshape(len(bundle.test_y_mask), -1).any(axis=1))
    )

    if len(train_x) == 0 or len(val_x) == 0:
        raise RuntimeError(
            f'{bundle.name}: no usable train/validation samples remain after target-mask filtering. '
            'Inspect the source NPZ and missing-value policy.'
        )

    print(
        f'[DATA] {bundle.name} supervised sample filter | '
        f'train kept={len(train_x)}/{len(bundle.train_x)} dropped={dropped_train} | '
        f'val kept={len(val_x)}/{len(bundle.val_x)} dropped={dropped_val} | '
        f'test kept={len(bundle.test_x)}/{len(bundle.test_x)} '
        f'(fully-invalid test rows={bundle.test_fully_invalid_samples}; preserved for timeline evaluation)',
        flush=True,
    )

    train = DataLoader(
        TrafficTensorDataset(train_x, train_y, train_m),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val = DataLoader(
        TrafficTensorDataset(val_x, val_y, val_m),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test = DataLoader(
        TrafficTensorDataset(bundle.test_x, bundle.test_y, bundle.test_y_mask),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train, val, test
