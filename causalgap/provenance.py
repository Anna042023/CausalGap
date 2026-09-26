import subprocess
from pathlib import Path


def git_commit(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
    except Exception:
        return None
