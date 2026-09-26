import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import parse, serve, silence_library_stdout


args = parse()
silence_library_stdout()

src = Path(args.root) / "class5" / "Moirai" / "uni2ts-main" / "src"
sys.path.insert(0, str(src))

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
module = MoiraiModule.from_pretrained(args.weights).to(DEVICE).eval()

PREDICTOR_CACHE = {}
PRED_BATCH = max(1, int(os.environ.get("CAUSALGAP_MOIRAI_PRED_BATCH", "64")))
REQUEST_CHUNK = max(
    PRED_BATCH,
    int(os.environ.get("CAUSALGAP_MOIRAI_REQUEST_CHUNK", "256")),
)


class CausalGapMoiraiForecast(MoiraiForecast):
    """Moirai forecast wrapper with paper-aligned pre-head representation export.

    The exported representation is the tensor immediately before Moirai's
    final distribution-parameter projection ``module.param_proj``. Only
    prediction-token states are pooled, using a fixed non-learned average.

    For patch_size='auto', official Moirai scores every candidate patch size
    and selects one independently for each sample. This wrapper preserves that
    prediction behavior and selects the representation from the same candidate
    that produced the chosen forecast.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causalgap_rep_sink = []

    def reset_causalgap_rep_sink(self):
        self._causalgap_rep_sink.clear()

    def consume_causalgap_rep_sink(self):
        if not self._causalgap_rep_sink:
            return None
        out = torch.cat(self._causalgap_rep_sink, dim=0)
        self._causalgap_rep_sink.clear()
        return out

    @staticmethod
    def _prediction_token_average(reprs, prediction_mask):
        """Fixed average pooling over prediction-token pre-head states."""
        if not torch.is_tensor(reprs):
            raise RuntimeError(
                "Moirai param_proj input is not a tensor; cannot export pre-head representation."
            )
        if reprs.ndim != 3:
            raise RuntimeError(
                f"Expected Moirai pre-head tensor [B,S,D], got shape={tuple(reprs.shape)}"
            )

        mask = prediction_mask.to(device=reprs.device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[:2] != reprs.shape[:2]:
            raise RuntimeError(
                "Moirai prediction-mask/pre-head shape mismatch: "
                f"mask={tuple(mask.shape)} repr={tuple(reprs.shape)}"
            )

        w = mask.to(dtype=reprs.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (reprs * w).sum(dim=1) / denom

    def _get_distr_and_prehead_rep(
        self,
        patch_size,
        past_target,
        past_observed_target,
        past_is_pad,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
    ):
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        captured = []

        def pre_head_hook(_module, inputs):
            # MoiraiModule.forward calls self.param_proj(reprs, patch_size),
            # therefore inputs[0] is exactly the pre-head representation.
            if inputs and torch.is_tensor(inputs[0]):
                captured.append(inputs[0].detach())

        handle = self.module.param_proj.register_forward_pre_hook(pre_head_hook)
        try:
            distr = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                torch.ones_like(time_id, dtype=torch.long) * patch_size,
            )
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(
                "Expected exactly one Moirai param_proj call in one _get_distr path, "
                f"captured {len(captured)}."
            )

        rep = self._prediction_token_average(captured[0], prediction_mask)
        return distr, rep

    def forward(
        self,
        past_target,
        past_observed_target,
        past_is_pad,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
        num_samples=None,
    ):
        if self.hparams.patch_size == "auto":
            val_loss = []
            preds = []
            reps = []

            for patch_size in self.module.patch_sizes:
                val_loss.append(
                    self._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : self.past_length, :],
                        observed_target=past_observed_target[..., : self.past_length, :],
                        is_pad=past_is_pad[..., : self.past_length],
                        feat_dynamic_real=(
                            feat_dynamic_real[..., : self.past_length, :]
                            if feat_dynamic_real is not None
                            else None
                        ),
                        observed_feat_dynamic_real=(
                            observed_feat_dynamic_real[..., : self.past_length, :]
                            if observed_feat_dynamic_real is not None
                            else None
                        ),
                        past_feat_dynamic_real=(
                            past_feat_dynamic_real[..., : self.hparams.context_length, :]
                            if past_feat_dynamic_real is not None
                            else None
                        ),
                        past_observed_feat_dynamic_real=(
                            past_observed_feat_dynamic_real[
                                ..., : self.hparams.context_length, :
                            ]
                            if past_observed_feat_dynamic_real is not None
                            else None
                        ),
                    )
                )

                distr, rep = self._get_distr_and_prehead_rep(
                    patch_size,
                    past_target[..., -self.hparams.context_length :, :],
                    past_observed_target[..., -self.hparams.context_length :, :],
                    past_is_pad[..., -self.hparams.context_length :],
                    (
                        feat_dynamic_real[..., -self.past_length :, :]
                        if feat_dynamic_real is not None
                        else None
                    ),
                    (
                        observed_feat_dynamic_real[..., -self.past_length :, :]
                        if observed_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_feat_dynamic_real[..., -self.hparams.context_length :, :]
                        if past_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_observed_feat_dynamic_real[
                            ..., -self.hparams.context_length :, :
                        ]
                        if past_observed_feat_dynamic_real is not None
                        else None
                    ),
                )

                preds.append(
                    self._format_preds(
                        patch_size,
                        distr.sample(torch.Size((num_samples or self.hparams.num_samples,))),
                        past_target.shape[-1],
                    )
                )
                reps.append(rep)

            val_loss = torch.stack(val_loss)  # [P,B]
            preds = torch.stack(preds)        # [P,B,S,H,*target]
            reps = torch.stack(reps)          # [P,B,D]

            idx = val_loss.argmin(dim=0)
            batch_idx = torch.arange(len(idx), device=idx.device)

            selected_pred = preds[idx, batch_idx]
            selected_rep = reps[idx, batch_idx]

            self._causalgap_rep_sink.append(selected_rep.detach().float().cpu())
            return selected_pred

        distr, rep = self._get_distr_and_prehead_rep(
            self.hparams.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real,
            observed_feat_dynamic_real,
            past_feat_dynamic_real,
            past_observed_feat_dynamic_real,
        )

        preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
        out = self._format_preds(
            self.hparams.patch_size,
            preds,
            past_target.shape[-1],
        )
        self._causalgap_rep_sink.append(rep.detach().float().cpu())
        return out


def get_predictor(H, num_samples, context):
    key = (int(H), int(num_samples), int(context), int(PRED_BATCH))
    if key not in PREDICTOR_CACHE:
        model = CausalGapMoiraiForecast(
            module=module,
            prediction_length=int(H),
            context_length=max(1, int(context)),
            patch_size="auto",
            num_samples=int(num_samples),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(DEVICE).eval()

        predictor = model.create_predictor(
            batch_size=PRED_BATCH,
            device=DEVICE,
        )
        PREDICTOR_CACHE[key] = (model, predictor)

    return PREDICTOR_CACHE[key]


def infer(flat, H, num_samples, need_rep=True, progress_cb=None):
    model, predictor = get_predictor(H, num_samples, flat.shape[1])
    pred_parts = []
    rep_parts = [] if need_rep else None
    total = len(flat)

    for s in range(0, total, REQUEST_CHUNK):
        chunk = np.ascontiguousarray(flat[s : s + REQUEST_CHUNK], dtype=np.float32)

        # Never allow stale representations to leak between requests.
        model.reset_causalgap_rep_sink()

        ds = ListDataset(
            [
                {
                    "start": pd.Period("2000-01-01", freq="h"),
                    "target": a,
                }
                for a in chunk
            ],
            freq="h",
        )

        mc_seed = 2027 + int(s)

        np.random.seed(mc_seed)
        torch.manual_seed(mc_seed)

        if torch.cuda.is_available():
          torch.cuda.manual_seed_all(mc_seed)

        with torch.inference_mode():
          fc = list(predictor.predict(ds))

        pred = np.stack(
            [np.median(f.samples, axis=0)[:H] for f in fc]
        ).astype(np.float32)
        pred_parts.append(pred)

        if need_rep:
            rep = model.consume_causalgap_rep_sink()
            if rep is None:
                raise RuntimeError("Moirai pre-head representation sink is empty.")
            if rep.shape[0] != len(chunk):
                raise RuntimeError(
                    "Moirai representation/sample alignment failure: "
                    f"rep_rows={rep.shape[0]} chunk_rows={len(chunk)}"
                )

            rep_np = rep.numpy().astype(np.float32, copy=False)
            if not np.isfinite(rep_np).all():
                raise RuntimeError("Moirai pre-head representation contains NaN/Inf.")
            rep_parts.append(rep_np)
        else:
            model.reset_causalgap_rep_sink()

        if progress_cb is not None:
            progress_cb(
                min(s + len(chunk), total),
                total,
                "moirai_chunk",
            )

        del fc, ds

    pred_all = np.concatenate(pred_parts, axis=0)

    if need_rep:
        rep_all = np.concatenate(rep_parts, axis=0)
        if rep_all.shape[0] != pred_all.shape[0]:
            raise RuntimeError(
                "Final Moirai prediction/representation row mismatch: "
                f"pred={pred_all.shape[0]} rep={rep_all.shape[0]}"
            )
    else:
        rep_all = None

    return pred_all, rep_all


serve(infer)
