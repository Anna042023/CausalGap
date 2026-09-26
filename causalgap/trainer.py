from __future__ import annotations

from pathlib import Path
import copy
import time
import numpy as np
import torch

from .utils import ensure_dir, save_json


MAE_LOSS_MODELS = {'DCRNN', 'GWNet', 'AGCRN'}


def _build_optimizer(model, lr, weight_decay, optimizer='auto'):
    name = str(optimizer or 'auto').lower()
    if name == 'auto':
        name = 'rmsprop' if getattr(model, 'model_name', '') == 'STGCN' else 'adam'
    if name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {optimizer}')


def _valid_mask(target, target_mask=None):
    valid = torch.isfinite(target)
    if target_mask is not None:
        m = target_mask.to(device=target.device, dtype=torch.bool)
        if m.shape != target.shape:
            raise ValueError(f'Target mask {tuple(m.shape)} != target {tuple(target.shape)}')
        valid = valid & m
    return valid


def _target_valid_count(target, target_mask=None) -> int:
    return int(_valid_mask(target, target_mask).sum().item())


def supervised_loss(pred, target, target_mask=None, policy='masked_mse'):
    """Unified target-validity handling with model-specific objective type.

    Missing speed targets are excluded through ``target_mask``.  A batch with no
    valid target is not a model error; such a batch has zero supervised signal and
    is skipped by ``train_model`` before the forward pass.  If this function still
    sees no valid element, that means valid targets existed but predictions became
    non-finite, which is treated as a numerical failure rather than silently hidden.
    """
    if pred.shape != target.shape:
        raise ValueError(f'Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}')
    policy = str(policy or 'masked_mse').lower()

    target_valid = _valid_mask(target, target_mask)
    if not torch.any(target_valid):
        raise RuntimeError('No valid target elements in this batch.')

    pred_finite = torch.isfinite(pred)
    if torch.any(target_valid & (~pred_finite)):
        bad = int((target_valid & (~pred_finite)).sum().item())
        raise RuntimeError(f'Model produced {bad} non-finite predictions at valid target positions.')

    valid = target_valid & pred_finite
    err = pred[valid] - target[valid]
    if policy in ('mse', 'masked_mse'):
        return torch.mean(err ** 2)
    if policy in ('mae', 'masked_mae'):
        return torch.mean(torch.abs(err))
    if policy == 'huber':
        return torch.nn.functional.smooth_l1_loss(pred[valid], target[valid])
    if policy == 'custom':
        raise RuntimeError('custom loss must be supplied by ModelOutput.loss')
    raise ValueError(f'Unknown loss_policy={policy}')


def _effective_loss_policy(model, requested):
    requested = str(requested or 'auto').lower()
    if requested != 'auto':
        return requested
    return 'masked_mae' if getattr(model, 'model_name', '') in MAE_LOSS_MODELS else 'masked_mse'


def _apply_epoch_lr_schedule(opt, initial_lr, epoch, schedule):
    schedule = str(schedule or 'none').lower()
    if schedule in ('none', 'stgcn_step'):
        return
    if schedule == 'tslib_type1':
        # Time-Series-Library type1: epoch 1 uses the initial LR; afterwards halve
        # it once per completed epoch.
        next_lr = float(initial_lr) * (0.5 ** max(0, int(epoch) - 1))
        for group in opt.param_groups:
            group['lr'] = next_lr
        return
    raise ValueError(f'Unknown lr_schedule={schedule}')


def _unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        mask = torch.ones_like(y, dtype=torch.bool)
    elif len(batch) == 3:
        x, y, mask = batch
    else:
        raise ValueError(f'Unexpected loader batch length={len(batch)}')
    return x, y, mask


def train_model(
    model,
    bundle,
    loaders,
    device,
    run_dir,
    epochs=50,
    lr=1e-3,
    patience=10,
    weight_decay=0.0,
    optimizer='auto',
    loss_policy='auto',
    lr_schedule='none',
):
    train_loader, val_loader, _ = loaders
    model = model.to(device)
    adj = torch.from_numpy(bundle.adj).float().to(device)
    opt = _build_optimizer(model, lr, weight_decay, optimizer)
    effective_loss = _effective_loss_policy(model, loss_policy)
    schedule = str(lr_schedule or 'none').lower()
    step_scheduler = (
        torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.7)
        if schedule == 'stgcn_step' else None
    )

    best = float('inf')
    best_state = None
    bad = 0
    best_epoch = None
    epochs_ran = 0
    total_skipped_train_batches = 0
    total_skipped_val_batches = 0
    run_dir = ensure_dir(run_dir)
    train_t0 = time.monotonic()
    print(
        f'Device={device} | optimizer={opt.__class__.__name__} | lr={lr:g} '
        f'| loss_policy={effective_loss} | lr_schedule={schedule}',
        flush=True,
    )

    initial_epoch_budget = int(epochs)
    cap_sensitive = getattr(model, 'model_name', '') in {'PatchTST','iTransformer','TimesNet','FEDformer','DLinear','Autoformer'}
    hard_epoch_limit = initial_epoch_budget + (max(int(patience), int(np.ceil(initial_epoch_budget * 0.5))) if cap_sensitive else 0)
    epoch_extensions_used = 0
    stop_reason = 'epoch_budget'
    ep = 0
    while ep < hard_epoch_limit:
        ep += 1
        if ep > initial_epoch_budget and epoch_extensions_used == 0:
            epoch_extensions_used = 1
            print(f'[AUTO-CONTINUE] {getattr(model, "model_name", "model")} reached initial epoch budget={initial_epoch_budget} without patience stop; extending to hard_limit={hard_epoch_limit}.', flush=True)
        epochs_ran = ep
        if hasattr(model, 'set_training_progress'):
            model.set_training_progress(ep, epochs)
        elif hasattr(model, 'base') and hasattr(model.base, 'set_training_progress'):
            model.base.set_training_progress(ep, epochs)

        model.train()
        loss_weighted_sum = 0.0
        loss_weight = 0
        custom_loss_values = []
        custom_loss_seen = False
        valid_elems_epoch = 0
        target_elems_epoch = 0
        skipped_train = 0

        for batch in train_loader:
            x, y, y_mask = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            # Some DCRNN-format METR-LA windows contain no observable target at all.
            # Such windows carry no supervised learning signal.  Skipping them is
            # mathematically equivalent to giving them zero weight, and prevents a
            # random shuffled batch from aborting every model's training run.
            target_valid = _valid_mask(y, y_mask)
            n_valid_target = int(target_valid.sum().item())
            target_elems_epoch += int(y.numel())
            valid_elems_epoch += n_valid_target
            if n_valid_target == 0:
                skipped_train += 1
                total_skipped_train_batches += 1
                continue

            opt.zero_grad(set_to_none=True)
            if hasattr(model, 'training_forward_with_mask'):
                out = model.training_forward_with_mask(x, adj, y, y_mask)
            else:
                out = model.training_forward(x, adj, y)
            target = y[:, :out.pred.size(1)]
            target_mask = y_mask[:, :out.pred.size(1)]
            n_valid = _target_valid_count(target, target_mask)
            if n_valid == 0:
                # Defensive guard for models with shorter prediction heads.
                skipped_train += 1
                total_skipped_train_batches += 1
                continue

            if getattr(out, 'loss', None) is not None:
                loss = out.loss
                custom_loss_seen = True
                custom_loss_values.append(float(loss.detach().item()))
            else:
                loss = supervised_loss(out.pred, target, target_mask, effective_loss)

            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {ep}: {loss}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # Weight epoch reporting by the number of valid supervised targets so
            # batches with many missing entries do not count as much as dense ones.
            loss_weighted_sum += float(loss.item()) * max(1, n_valid)
            loss_weight += max(1, n_valid)

        if loss_weight == 0:
            raise RuntimeError(
                'Training split contains no usable supervised targets after masking. '
                'Check the dataset/missing-value policy instead of disabling masking.'
            )

        # Validation checkpoint selection uses one globally weighted masked MSE.
        model.eval()
        val_sse = 0.0
        val_count = 0
        skipped_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, y_mask = _unpack_batch(batch)
                x = x.to(device)
                y = y.to(device)
                y_mask = y_mask.to(device)
                target_valid = _valid_mask(y, y_mask)
                if not torch.any(target_valid):
                    skipped_val += 1
                    total_skipped_val_batches += 1
                    continue

                out = model.inference_forward(x, adj, bundle.pred_len)
                target = y[:, :out.pred.size(1)]
                target_mask = y_mask[:, :out.pred.size(1)]
                valid = _valid_mask(target, target_mask)
                if not torch.any(valid):
                    skipped_val += 1
                    total_skipped_val_batches += 1
                    continue
                if torch.any(valid & (~torch.isfinite(out.pred))):
                    n_bad = int((valid & (~torch.isfinite(out.pred))).sum().item())
                    raise RuntimeError(f'Model produced {n_bad} non-finite validation predictions.')
                err = out.pred[valid] - target[valid]
                val_sse += float(torch.sum(err ** 2).item())
                val_count += int(valid.sum().item())

        if val_count == 0:
            raise RuntimeError(
                'Validation split contains no usable targets after masking. '
                'Check METR-LA/PEMS-BAY missing-value preprocessing.'
            )

        tr = float(loss_weighted_sum / loss_weight)
        va = float(val_sse / val_count)
        current_lr = float(opt.param_groups[0]['lr'])
        train_label = 'custom_total' if custom_loss_seen else effective_loss
        valid_fraction = valid_elems_epoch / max(1, target_elems_epoch)
        print(
            f'Epoch {ep:03d} | train {train_label} {tr:.6f} | '
            f'val masked_mse {va:.6f} | lr {current_lr:.6g} | '
            f'valid_target={valid_fraction:.2%} | '
            f'skipped_empty_train_batches={skipped_train} | skipped_empty_val_batches={skipped_val}',
            flush=True,
        )

        if va < best - 1e-8:
            best = va
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            torch.save({
                'model': best_state,
                'best_val': best,
                'epoch': ep,
                'data_signature': {
                    'dataset': bundle.name,
                    'history_len': bundle.history_len,
                    'pred_len': bundle.pred_len,
                    'num_nodes': bundle.num_nodes,
                    'scaler_mean': float(bundle.scaler.mean),
                    'scaler_std': float(bundle.scaler.std),
                    'scaler_fit': bundle.scaler_fit,
                    'split_protocol': bundle.split_protocol,
                    'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
                    'coordinate_source': str(getattr(bundle, 'coordinate_source', 'missing_zero_filled')),
                    'model_protocol': str(getattr(model, 'model_protocol', 'v3.3-native')),
                    'forecast_protocol': 'direct_horizon_matched',
                    'output_space': str(getattr(model, 'output_space', 'normalized')),
                    'missing_value_policy': str(bundle.missing_value_policy),
                    'speed_missing_input_fill': str(bundle.speed_missing_input_fill),
                    'training_loss_policy': effective_loss,
                    'metric_protocol': str(bundle.metric_protocol),
                    'empty_target_batch_policy': 'skip_zero_weight',
                },
            }, Path(run_dir) / 'checkpoint.pt')
        else:
            bad += 1

        if step_scheduler is not None:
            step_scheduler.step()
        else:
            _apply_epoch_lr_schedule(opt, lr, ep, schedule)

        if bad >= int(patience):
            stop_reason = 'early_stopping'
            print(f'Early stopping at epoch {ep}, best val={best:.6f}', flush=True)
            break

        if ep >= initial_epoch_budget and not cap_sensitive:
            stop_reason = 'epoch_budget'
            break
        if ep >= hard_epoch_limit:
            stop_reason = 'hard_epoch_limit'
            break

    if best_state is None:
        raise RuntimeError('Training finished without a valid checkpoint.')
    model.load_state_dict(best_state)

    training_seconds = float(time.monotonic() - train_t0)
    save_json(Path(run_dir) / 'training_summary.json', {
        'model': str(getattr(model, 'model_name', 'unknown')),
        'dataset': bundle.name,
        'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
        'epochs_requested': int(initial_epoch_budget),
        'initial_epoch_budget': int(initial_epoch_budget),
        'hard_epoch_limit': int(hard_epoch_limit),
        'epoch_extensions_used': int(epoch_extensions_used),
        'stop_reason': stop_reason,
        'lr_schedule_version': 'source_aligned_v3.3.3',
        'epochs_ran': int(epochs_ran),
        'best_epoch': (int(best_epoch) if best_epoch is not None else None),
        'best_val_loss': (float(best) if np.isfinite(best) else None),
        'best_val_mse': (float(best) if np.isfinite(best) else None),
        'validation_loss_definition': 'masked_mse',
        'training_seconds': training_seconds,
        'optimizer': opt.__class__.__name__,
        'initial_lr': float(lr),
        'weight_decay': float(weight_decay),
        'early_stopping_patience': int(patience),
        'training_loss_policy': effective_loss,
        'lr_schedule': schedule,
        'missing_value_policy': str(bundle.missing_value_policy),
        'metric_protocol': str(bundle.metric_protocol),
        'empty_target_batch_policy': 'skip_zero_weight',
        'skipped_empty_train_batches_total': int(total_skipped_train_batches),
        'skipped_empty_val_batches_total': int(total_skipped_val_batches),
    })
    print(
        f'Training finished in {training_seconds/60:.2f} min | '
        f'best_epoch={best_epoch} | best_val={best:.6f} | '
        f'skipped_empty_train_batches={total_skipped_train_batches} | '
        f'skipped_empty_val_batches={total_skipped_val_batches}',
        flush=True,
    )
    return model
