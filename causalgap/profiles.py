"""Formal/smoke defaults for the 21-model CausalGap benchmark.

v3.3 keeps one unified interface but no longer forces one optimization recipe on
all architectures.  Each baseline has a small model-specific profile; every
resolved value is written to run_config.json and remains CLI-overridable.
"""

BASE = dict(
    ks=3, kt=3, dropout=0.1, d_model=64, d_ff=128,
    e_layers=2, d_layers=1, n_heads=4, moving_avg=3, factor=1,
    patience=10, weight_decay=0.0, optimizer='auto',
    loss_policy='masked_mse', lr_schedule='none',
)


def p(batch, epochs, lr, **kw):
    d = dict(BASE)
    d.update(batch_size=batch, epochs=epochs, lr=lr)
    d.update(kw)
    return d


PAPER = {
    'STGCN': p(50, 50, 1e-3, dropout=0.0, optimizer='rmsprop', loss_policy='masked_mse', lr_schedule='stgcn_step'),
    'DCRNN': p(64, 100, 1e-2, d_model=64, e_layers=2, optimizer='adam', loss_policy='masked_mae'),
    'ASTGNN': p(32, 50, 1e-3, d_model=64, e_layers=3, n_heads=4, optimizer='adam', loss_policy='masked_mse'),
    'GWNet': p(64, 50, 1e-3, dropout=.3, optimizer='adam', loss_policy='masked_mae'),
    'AGCRN': p(64, 100, 3e-3, d_model=64, e_layers=2, optimizer='adam', loss_policy='masked_mae'),

    # Traffic-only STEVE adaptation: give the learned invariant/variant gate time
    # to converge; the old fixed 0.05 shortcut is removed in v3.3.
    'STEVE': p(32, 50, 1e-3, d_model=64, patience=8, optimizer='adam', loss_policy='masked_mae'),
    'DCGCN': p(32, 50, 1e-3, d_model=64, e_layers=2, optimizer='adam', loss_policy='masked_mse'),
    'ST-Norm': p(32, 50, 1e-3, d_model=32, optimizer='adam', loss_policy='masked_mse'),

    'CCHMM': p(32, 100, 5e-4, d_model=64, optimizer='adam', loss_policy='masked_mse'),
    'CausalGRIT': p(32, 60, 1e-3, d_model=64, e_layers=1, dropout=.1, optimizer='adam', loss_policy='custom'),
    'TF-MVGNN': p(32, 50, 1e-3, d_model=64, e_layers=1, dropout=0.0, optimizer='adam', loss_policy='masked_mse'),

    # Time-Series-Library adapters preserve node-as-variable [B,L,N] input.
    # Short-window-specific architectural safety is handled inside TSLibModel.
    'PatchTST': p(16, 10, 1e-4, d_model=512, d_ff=512, e_layers=2, n_heads=8,
                  moving_avg=25, factor=3, patience=3, optimizer='adam', loss_policy='masked_mse', lr_schedule='none'),
    'iTransformer': p(16, 20, 1e-3, d_model=512, d_ff=512, e_layers=4, n_heads=8,
                      moving_avg=25, factor=3, patience=5, optimizer='adam', loss_policy='masked_mse', lr_schedule='tslib_type1'),
    'TimesNet': p(32, 10, 1e-4, d_model=512, d_ff=512, e_layers=2, n_heads=8,
                  moving_avg=25, factor=3, patience=3, optimizer='adam', loss_policy='masked_mse', lr_schedule='tslib_type1'),
    'FEDformer': p(32, 15, 1e-4, d_model=512, d_ff=2048, e_layers=2, n_heads=8,
                   moving_avg=25, factor=3, patience=4, optimizer='adam', loss_policy='masked_mse', lr_schedule='tslib_type1'),
    # DLinear is cheap and the old 10-epoch cap was repeatedly hit. 30 epochs +
    # early stopping prevents truncating an improving run.
    'DLinear': p(32, 30, 1e-4, d_model=512, d_ff=2048, e_layers=2, n_heads=8,
                 moving_avg=25, factor=1, patience=5, optimizer='adam', loss_policy='masked_mse', lr_schedule='none'),
    # Old unified profile accidentally used 3 as total epochs. v3.3 restores a
    # 10-epoch budget and uses patience=3 as the early-stopping control.
    'Autoformer': p(32, 10, 1e-4, d_model=512, d_ff=2048, e_layers=2, n_heads=8,
                    moving_avg=25, factor=3, patience=3, optimizer='adam', loss_policy='masked_mse', lr_schedule='tslib_type1'),

    'Chronos': p(16, 0, 0.0),
    'TimesFM': p(16, 0, 0.0),
    'Moirai': p(16, 0, 0.0),
    'Lag-Llama': p(16, 0, 0.0),
}

SMOKE = {
    m: p(8, 2, 1e-3, d_model=32, d_ff=64, e_layers=1, n_heads=4,
         patience=2, loss_policy='masked_mse')
    for m in PAPER
}
for m in ('Chronos', 'TimesFM', 'Moirai', 'Lag-Llama'):
    SMOKE[m].update(epochs=0, lr=0.0, batch_size=4)


def _safe_odd_kernel(seq_len: int, requested: int) -> int:
    seq_len = max(1, int(seq_len))
    requested = max(1, int(requested))
    if requested <= seq_len and requested % 2 == 1:
        return requested
    k = min(requested, seq_len)
    if k % 2 == 0:
        k -= 1
    return max(1, k)


def resolve(args, model_name, history_len=None, num_nodes=None):
    cfg = (PAPER if getattr(args, 'profile', 'paper') == 'paper' else SMOKE)[model_name]
    for k, v in cfg.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    if num_nodes is not None:
        n = int(num_nodes)
        if model_name in ('Chronos', 'TimesFM', 'Moirai', 'Lag-Llama') and getattr(args, 'profile', 'paper') == 'paper':
            if n >= 1000:
                args.batch_size = min(int(args.batch_size), 4)
            elif n >= 500:
                args.batch_size = min(int(args.batch_size), 8)
            else:
                args.batch_size = min(int(args.batch_size), 16)
        elif n >= 1000:
            heavy = {
                'STGCN', 'DCRNN', 'ASTGNN', 'GWNet', 'AGCRN', 'STEVE',
                'DCGCN', 'CCHMM', 'CausalGRIT', 'TF-MVGNN'
            }
            cap = 4 if model_name in heavy else 8
            args.batch_size = min(int(args.batch_size), cap)

    if model_name == 'PatchTST' and history_len is not None:
        if getattr(args, 'patch_len', None) is None:
            args.patch_len = min(16, max(2, int(history_len) // 3))
        if getattr(args, 'stride', None) is None:
            args.stride = max(1, int(args.patch_len) // 2)

    if model_name in ('FEDformer', 'Autoformer', 'DLinear') and history_len is not None:
        args.moving_avg = _safe_odd_kernel(int(history_len), int(args.moving_avg))

    return args
