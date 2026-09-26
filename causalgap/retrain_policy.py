from __future__ import annotations

"""Central Clean-Final retraining policy for CausalGap v3.3.3."""
from .models.registry import SPECS

SPEED_DATASETS = {'METR-LA', 'PEMS-BAY'}
LOSS_CHANGED_V33 = {'DCRNN', 'GWNet', 'AGCRN'}
TSLIB_LR_FIXED_V333 = {'iTransformer', 'TimesNet', 'FEDformer', 'Autoformer'}
ARCH_CHANGED_V333 = {'STEVE'}
CAP_SENSITIVE = {'PatchTST','iTransformer','TimesNet','FEDformer','DLinear','Autoformer'}

def _is_foundation(model):
    return SPECS[model].training_mode == 'zero_shot'

def _code(cfg):
    return str((cfg or {}).get('code_version', ''))

def training_protocol_is_final(dataset, model, run_config=None, training_summary=None):
    if _is_foundation(model): return True
    cfg, tr = run_config or {}, training_summary or {}
    if not _code(cfg).startswith('3.3.3'): return False
    if str(tr.get('validation_loss_definition', '')) != 'masked_mse': return False
    if dataset in SPEED_DATASETS and 'masked_zero' not in str(tr.get('missing_value_policy', '')): return False
    if model in LOSS_CHANGED_V33 and str(tr.get('training_loss_policy', '')) != 'masked_mae': return False
    if model in TSLIB_LR_FIXED_V333 and str(tr.get('lr_schedule_version', '')) != 'source_aligned_v3.3.3': return False
    if model == 'STEVE' and 'steve-released-core-traffic-only' not in str(cfg.get('model_protocol', '')): return False
    return True

def requires_v33_retraining(dataset, model, run_config=None, training_summary=None):
    """Backward-compatible name used by existing scripts."""
    if _is_foundation(model): return False
    cfg, tr = run_config or {}, training_summary or {}
    if training_protocol_is_final(dataset, model, cfg, tr): return False
    # Only invalidate learned parameters when the scientific training contract changed.
    if dataset in SPEED_DATASETS and not _code(cfg).startswith('3.3'):
        return True
    if model in LOSS_CHANGED_V33 and str(tr.get('training_loss_policy', '')) != 'masked_mae':
        return True
    if model in TSLIB_LR_FIXED_V333:
        return True
    if model in ARCH_CHANGED_V333:
        return True
    # v3.3+ nonaffected models remain valid even if metadata is older than 3.3.3.
    return False

def requires_finalization(dataset, model, run_config=None, training_summary=None):
    if requires_v33_retraining(dataset, model, run_config, training_summary):
        return True
    if _is_foundation(model):
        return False
    tr = training_summary or {}
    if model in CAP_SENSITIVE:
        try:
            ran = int(tr.get('epochs_ran'))
            best = int(tr.get('best_epoch'))
            stop = str(tr.get('stop_reason',''))
            # Legacy run that selected its best checkpoint at the cap is unresolved.
            if best >= ran and stop != 'early_stopping':
                return True
        except Exception:
            pass
    return False

ARCH_OR_PROFILE_CHANGED_V33 = set(TSLIB_LR_FIXED_V333) | {'STEVE','PatchTST','DLinear'}
