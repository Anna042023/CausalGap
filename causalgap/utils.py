import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def device_from_arg(device: str):
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA requested but unavailable, falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device)
