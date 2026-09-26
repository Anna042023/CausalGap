import argparse
import json
import sys
import traceback

import numpy as np

_PROTOCOL_STDOUT = sys.stdout


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--root', required=True)
    return ap.parse_args()


def silence_library_stdout():
    sys.stdout = sys.stderr


def protocol_send(obj):
    _PROTOCOL_STDOUT.write(json.dumps(obj) + '\n')
    _PROTOCOL_STDOUT.flush()


def flatten_x(x):
    # B,T,N,1 -> BN,T
    B, T, N, _ = x.shape
    return x[..., 0].transpose(0, 2, 1).reshape(B * N, T), B, N


def save_response(path, pred_flat, rep_flat, B, N, H):
    pred = np.asarray(pred_flat, dtype=np.float32).reshape(B, N, H).transpose(0, 2, 1)[..., None]
    if rep_flat is None:
        rep = np.empty((B, N, 0), dtype=np.float32)
    else:
        rep = np.asarray(rep_flat, dtype=np.float32).reshape(B, N, -1)
    with open(path, 'wb') as f:
        np.savez(f, pred=pred, rep=rep)


def serve(infer_fn):
    protocol_send({'ready': True})
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            cmd = msg.get('cmd')
            if cmd == 'close':
                break
            if cmd != 'infer':
                raise ValueError(f'unknown cmd {cmd}')
            z = np.load(msg['request'], allow_pickle=False)
            x = z['x']
            H = int(z['pred_len'])
            ns = int(z['num_samples'])
            need_rep = bool(int(z['need_rep'])) if 'need_rep' in z.files else True
            flat, B, N = flatten_x(x)

            def progress(done, total, stage='infer'):
                protocol_send({
                    'progress': True,
                    'stage': str(stage),
                    'done': int(done),
                    'total': int(total),
                })

            pred, rep = infer_fn(flat, H, ns, need_rep, progress)
            save_response(msg['response'], pred, rep, B, N, H)
            protocol_send({'ok': True})
        except Exception as e:
            protocol_send({
                'ok': False,
                'error': repr(e),
                'traceback': traceback.format_exc(),
            })
