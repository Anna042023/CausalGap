import atexit
import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from ..base import UnifiedModel, ModelOutput
from ...config import PROJECT_ROOT

ENV_MAP = {
    'Chronos': Path('/data/wanganna/envs/chronos/bin/python'),
    'TimesFM': Path('/data/wanganna/envs/timesfm/bin/python'),
    'Moirai': Path('/data/wanganna/envs/moirai/bin/python'),
    'Lag-Llama': Path('/data/wanganna/envs/lagllama/bin/python'),
}
WEIGHT_MAP = {
    'Chronos': PROJECT_ROOT / 'model_weights' / 'Chronos',
    'TimesFM': PROJECT_ROOT / 'model_weights' / 'TimesFM',
    'Moirai': PROJECT_ROOT / 'model_weights' / 'Moirai',
    'Lag-Llama': PROJECT_ROOT / 'model_weights' / 'Lag-Llama',
}


class FoundationSubprocessModel(UnifiedModel):
    """Zero-shot foundation-model bridge with persistent worker + resumable cache.

    v3.2.1 keeps the scientific protocol unchanged and only hardens execution:
      * worker progress messages turn the inference timeout into an *inactivity*
        timeout instead of a total-request timeout;
      * a stuck worker is killed and restarted cleanly;
      * the current outer batch is recursively split B -> B/2 -> ... -> 1 when
        a worker times out, so one pathological batch cannot kill a full test;
      * split decisions are written next to the content-addressed cache, so an
        interrupted rerun does not repeatedly retry the same known-bad batch;
      * successful split results are reassembled and cached under the original
        full-batch key, preserving resume behavior.

    Predictions/representations are unchanged relative to v3.2: workers receive
    the same raw contexts and num_samples, and probabilistic models still return
    the sample median as the point forecast.
    """

    uses_graph = False
    is_zero_shot = True
    model_protocol = 'v3.2.1-foundation-zero-shot-batched-cached-recoverable-worker'

    def __init__(
        self,
        model_name,
        num_nodes,
        history_len,
        pred_len,
        interval_minutes=5,
        env_python=None,
        weights=None,
        num_samples=100,
        scaler_mean=0.0,
        scaler_std=1.0,
        worker_startup_timeout=180,
        worker_infer_timeout=600,
        adaptive_split=True,
        max_split_depth=8,
        single_window_retries=1,
        cache_enabled=False,
        cache_dir=None,
        cache_readonly=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_nodes = int(num_nodes)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.interval_minutes = int(interval_minutes)
        self.num_samples = int(num_samples)
        self.scaler_mean = float(scaler_mean)
        self.scaler_std = float(scaler_std)
        self.worker_startup_timeout = int(worker_startup_timeout)
        # IMPORTANT: from v3.2.1 this is an inactivity timeout. Progress heartbeats
        # from the worker reset the deadline, so a healthy long request may run
        # longer than this value without being killed.
        self.worker_infer_timeout = int(worker_infer_timeout)
        self.adaptive_split = bool(adaptive_split)
        self.max_split_depth = max(0, int(max_split_depth))
        self.single_window_retries = max(0, int(single_window_retries))
        self.env_python = Path(env_python or ENV_MAP[model_name])
        self.weights = Path(weights or WEIGHT_MAP[model_name])
        self.worker = (
            Path(__file__).resolve().parents[3]
            / 'foundation_workers'
            / f"{model_name.lower().replace('-', '_')}_worker.py"
        )
        self._proc = None
        self._tmp = Path(tempfile.mkdtemp(prefix=f'causalgap_{model_name.lower()}_'))
        self.cache_enabled = bool(cache_enabled)
        self.cache_readonly = bool(cache_readonly)
        self.cache_dir = Path(cache_dir) if cache_dir else (PROJECT_ROOT / 'cache' / 'foundation_inference')
        self.cache_dir = self.cache_dir / self.model_name.replace('/', '_')
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self.timeout_recoveries = 0
        self.adaptive_splits = 0
        self.worker_restarts = 0
        self.progress_messages = 0
        self._weight_signature = self._make_weight_signature()
        atexit.register(self.close)

    @property
    def representation_hook_name(self):
        if self.model_name == 'Moirai':
            return 'Moirai_postfusion_prediction_token_pre_param_proj_fixed_average'
        return f'{self.model_name}_official_internal_encoder_hidden'

    def _make_weight_signature(self):
        h = hashlib.blake2b(digest_size=12)
        h.update(str(self.weights.resolve()).encode())
        if self.weights.is_file():
            items = [self.weights]
        elif self.weights.exists():
            items = sorted(p for p in self.weights.rglob('*') if p.is_file())
        else:
            items = []
        for p in items:
            try:
                st = p.stat()
                h.update(str(p.relative_to(self.weights) if self.weights.is_dir() else p.name).encode())
                h.update(str(st.st_size).encode())
                h.update(str(st.st_mtime_ns).encode())
            except OSError:
                continue
        return h.hexdigest()

    def cache_stats(self):
        return {
            'foundation_cache_enabled': bool(self.cache_enabled),
            'foundation_cache_readonly': bool(self.cache_readonly),
            'foundation_cache_hits': int(self.cache_hits),
            'foundation_cache_misses': int(self.cache_misses),
            'foundation_cache_writes': int(self.cache_writes),
            'foundation_cache_dir': str(self.cache_dir),
            'foundation_worker_inactivity_timeout_s': int(self.worker_infer_timeout),
            'foundation_timeout_recoveries': int(self.timeout_recoveries),
            'foundation_adaptive_splits': int(self.adaptive_splits),
            'foundation_worker_restarts': int(self.worker_restarts),
            'foundation_progress_messages': int(self.progress_messages),
        }

    def _cache_path_raw(self, x_raw, need_rep):
        h = hashlib.blake2b(digest_size=20)
        header = {
            # Keep cache_version=v3.2 intentionally. The numerical protocol and
            # cache payload are unchanged, so already-computed v3.2 batches remain
            # valid and are reused by v3.2.1.
            'cache_version': 'v3.2',
            'model': self.model_name,
            'weights': self._weight_signature,
            'pred_len': self.pred_len,
            'interval_minutes': self.interval_minutes,
            'num_samples': self.num_samples,
            'scaler_mean': round(self.scaler_mean, 10),
            'scaler_std': round(self.scaler_std, 10),
            'need_rep': bool(need_rep),
            'shape': list(x_raw.shape),
        }
        h.update(json.dumps(header, sort_keys=True).encode())
        a = np.ascontiguousarray(x_raw, dtype=np.float32)
        h.update(memoryview(a).cast('B'))
        return self.cache_dir / f'{h.hexdigest()}.npz'

    def _cache_path(self, x_raw, need_rep):
        return self._cache_path_raw(x_raw, need_rep)

    @staticmethod
    def _split_marker(cache_path):
        return cache_path.with_suffix('.split.json') if cache_path is not None else None

    def _read_cache(self, cache_path):
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                pred_raw = np.asarray(z['pred_raw'], dtype=np.float32)
                rep_raw = np.asarray(z['rep'], dtype=np.float32)
            self.cache_hits += 1
            return pred_raw, rep_raw
        except Exception as e:
            print(f'[WARN] Ignoring unreadable foundation cache {cache_path}: {e}', file=sys.stderr, flush=True)
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

    def _write_cache(self, cache_path, pred_raw, rep_raw):
        if cache_path is None or self.cache_readonly:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            with tmp.open('wb') as f:
                np.savez(
                    f,
                    pred_raw=np.asarray(pred_raw, dtype=np.float32),
                    rep=np.asarray(rep_raw, dtype=np.float32),
                )
            os.replace(tmp, cache_path)
            self.cache_writes += 1
        except Exception as e:
            print(f'[WARN] Failed to write foundation cache {cache_path}: {e}', file=sys.stderr, flush=True)

    def _write_split_marker(self, marker, batch_size, depth, reason):
        if marker is None or self.cache_readonly:
            return
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            tmp = marker.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            tmp.write_text(json.dumps({
                'model': self.model_name,
                'batch_size': int(batch_size),
                'depth': int(depth),
                'reason': str(reason),
                'created_at_unix': time.time(),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(tmp, marker)
        except Exception:
            pass

    @staticmethod
    def _remove_marker(marker):
        if marker is not None:
            try:
                marker.unlink()
            except OSError:
                pass

    def _read_protocol_json(self, timeout, stage):
        """Read protocol JSON with an inactivity deadline.

        A worker may emit {"progress": true, ...}. Such a message proves the
        current request is alive and resets the inactivity deadline. This prevents
        healthy large requests from being killed merely because their *total*
        runtime exceeds timeout seconds.
        """
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError(f'{self.model_name} worker is not running.')
        inactivity = float(timeout)
        deadline = time.monotonic() + inactivity
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'{self.model_name} worker timed out during {stage} after '
                    f'{timeout}s without progress.'
                )
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                continue
            line = self._proc.stdout.readline()
            if line == '':
                code = self._proc.poll()
                raise RuntimeError(
                    f'{self.model_name} worker exited during {stage}; return code={code}.'
                )
            text = line.strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                print(f'[{self.model_name} worker log] {text}', file=sys.stderr, flush=True)
                deadline = time.monotonic() + inactivity
                continue
            if bool(msg.get('progress', False)):
                self.progress_messages += 1
                deadline = time.monotonic() + inactivity
                continue
            return msg

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.env_python.exists():
            raise RuntimeError(
                f'{self.model_name} environment missing: {self.env_python}. '
                f'Run scripts/env/install_{self.model_name.lower().replace("-", "_")}_env.sh'
            )
        if not self.worker.exists():
            raise RuntimeError(f'Worker missing: {self.worker}')

        cmd = [
            str(self.env_python), '-u', str(self.worker),
            '--weights', str(self.weights), '--root', str(PROJECT_ROOT),
        ]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        msg = self._read_protocol_json(self.worker_startup_timeout, 'startup')
        if not msg.get('ready'):
            raise RuntimeError(f'{self.model_name} worker startup failed: {msg}')

    def _hard_reset_worker(self):
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
        self.worker_restarts += 1

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({'cmd': 'close'}) + '\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    @staticmethod
    def _to_output(pred_raw, rep_raw, x, scaler_mean, scaler_std):
        pred_norm = (pred_raw - scaler_mean) / max(scaler_std, 1e-8)
        pred = torch.from_numpy(np.asarray(pred_norm, dtype=np.float32)).to(x.device, x.dtype)
        rep = torch.from_numpy(np.asarray(rep_raw, dtype=np.float32)).to(x.device, x.dtype)
        return ModelOutput(pred, rep)

    def _worker_infer_raw(self, x_raw, need_rep):
        """One worker request. Cache and split logic live outside this method."""
        self._ensure()
        token = uuid.uuid4().hex
        req = self._tmp / f'{token}_req.npz'
        resp = self._tmp / f'{token}_resp.npz'
        np.savez(
            req,
            x=np.asarray(x_raw, dtype=np.float32),
            pred_len=np.int64(self.pred_len),
            interval_minutes=np.int64(self.interval_minutes),
            num_samples=np.int64(self.num_samples),
            need_rep=np.int8(1 if need_rep else 0),
        )
        try:
            self._proc.stdin.write(
                json.dumps({'cmd': 'infer', 'request': str(req), 'response': str(resp)}) + '\n'
            )
            self._proc.stdin.flush()
            msg = self._read_protocol_json(self.worker_infer_timeout, 'inference')
            if not msg.get('ok'):
                raise RuntimeError(
                    f'{self.model_name} inference failed: {msg.get("error", msg)}\n'
                    f'{msg.get("traceback", "")}'
                )
            if not resp.exists():
                raise RuntimeError(
                    f'{self.model_name} worker reported success but response is missing: {resp}'
                )
            with np.load(resp, allow_pickle=False) as z:
                pred_raw = np.asarray(z['pred'], dtype=np.float32)
                rep_raw = np.asarray(z['rep'], dtype=np.float32)
            return pred_raw, rep_raw
        finally:
            for p in (req, resp):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _infer_raw_adaptive(self, x_raw, need_rep, depth=0):
        x_raw = np.ascontiguousarray(x_raw, dtype=np.float32)
        cache_path = self._cache_path_raw(x_raw, need_rep) if self.cache_enabled else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        self.cache_misses += 1
        marker = self._split_marker(cache_path)
        known_split = bool(marker is not None and marker.exists())

        # If a previous run already discovered that this exact content-addressed
        # batch stalls, immediately reuse the split plan instead of wasting another
        # timeout interval.
        if known_split and len(x_raw) > 1 and self.adaptive_split:
            return self._split_and_infer(x_raw, need_rep, depth, cache_path, marker, 'persisted split marker')

        try:
            pred_raw, rep_raw = self._worker_infer_raw(x_raw, need_rep)
            self._write_cache(cache_path, pred_raw, rep_raw)
            self._remove_marker(marker)
            return pred_raw, rep_raw
        except TimeoutError as e:
            self.timeout_recoveries += 1
            print(
                f'[RECOVER] {self.model_name} inactivity timeout on outer_batch={len(x_raw)} '
                f'depth={depth}: {e}',
                file=sys.stderr, flush=True,
            )
            self._hard_reset_worker()
            self._write_split_marker(marker, len(x_raw), depth, 'timeout')
            if self.adaptive_split and len(x_raw) > 1 and depth < self.max_split_depth:
                return self._split_and_infer(x_raw, need_rep, depth, cache_path, marker, 'timeout')

            # B==1 cannot be split at the outer-window level. Worker-level series
            # micro-batching should normally prevent this; allow a bounded clean
            # restart retry before finally surfacing the error.
            last = e
            for retry in range(self.single_window_retries):
                print(
                    f'[RECOVER] {self.model_name} retrying single outer window '
                    f'after worker restart ({retry+1}/{self.single_window_retries}).',
                    file=sys.stderr, flush=True,
                )
                try:
                    pred_raw, rep_raw = self._worker_infer_raw(x_raw, need_rep)
                    self._write_cache(cache_path, pred_raw, rep_raw)
                    self._remove_marker(marker)
                    return pred_raw, rep_raw
                except TimeoutError as e2:
                    last = e2
                    self.timeout_recoveries += 1
                    self._hard_reset_worker()
            raise last
        except (BrokenPipeError, ConnectionResetError) as e:
            self._hard_reset_worker()
            if self.adaptive_split and len(x_raw) > 1 and depth < self.max_split_depth:
                self._write_split_marker(marker, len(x_raw), depth, type(e).__name__)
                return self._split_and_infer(x_raw, need_rep, depth, cache_path, marker, type(e).__name__)
            raise

    def _split_and_infer(self, x_raw, need_rep, depth, cache_path, marker, reason):
        if len(x_raw) <= 1:
            raise RuntimeError('Adaptive split requested for a single outer window.')
        mid = len(x_raw) // 2
        self.adaptive_splits += 1
        print(
            f'[RECOVER] {self.model_name} splitting outer batch {len(x_raw)} -> '
            f'{mid}+{len(x_raw)-mid} ({reason}).',
            file=sys.stderr, flush=True,
        )
        p1, r1 = self._infer_raw_adaptive(x_raw[:mid], need_rep, depth + 1)
        p2, r2 = self._infer_raw_adaptive(x_raw[mid:], need_rep, depth + 1)
        pred_raw = np.concatenate([p1, p2], axis=0)
        rep_raw = np.concatenate([r1, r2], axis=0)
        # Write a composite cache entry under the original full-batch key. On the
        # next run this batch becomes an immediate cache hit.
        self._write_cache(cache_path, pred_raw, rep_raw)
        self._remove_marker(marker)
        return pred_raw, rep_raw

    def _infer(self, x, need_rep=True):
        x_raw = (
            x.detach().cpu().numpy().astype(np.float32) * self.scaler_std
            + self.scaler_mean
        )
        pred_raw, rep_raw = self._infer_raw_adaptive(x_raw, bool(need_rep), depth=0)
        return self._to_output(pred_raw, rep_raw, x, self.scaler_mean, self.scaler_std)

    def training_forward(self, x, adj, y=None):
        raise RuntimeError(
            f'{self.model_name} is zero-shot in the main benchmark; use `run.py evaluate`, not train.'
        )

    def inference_forward(self, x, adj, pred_len):
        return self.inference_forward_selective_rep(x, adj, pred_len, need_rep=True)

    def inference_forward_selective_rep(self, x, adj, pred_len, need_rep=True):
        if pred_len != self.pred_len:
            raise ValueError((pred_len, self.pred_len))
        return self._infer(x, need_rep=bool(need_rep))
