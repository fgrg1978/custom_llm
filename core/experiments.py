"""
Helper de tracking de experimentos.

Cada run de evaluate/tournament/train vuelca un directorio
experiments/{timestamp}_{name}/ con:
  - config.json : args usados
  - result.json : metricas devueltas
  - meta.json   : git SHA, branch, dirty flag, timestamp

Diseño: sin dependencias externas (no W&B todavia) pero con estructura
suficiente para hacer bisect de regresiones y reproducir runs.
"""

import os
import json
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")


def _run_git(args):
    try:
        return subprocess.check_output(
            ["/usr/bin/git"] + args,
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _git_sha():
    return _run_git(["rev-parse", "HEAD"]) or "unknown"


def _git_branch():
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def _git_dirty():
    out = _run_git(["status", "--porcelain"])
    if out is None:
        return None
    return bool(out)


def _jsonify(obj):
    """Convierte recursivamente a tipos serializables a JSON."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def log_experiment(name, config, result):
    """
    Persiste un experimento en experiments/{timestamp}_{name}/.

    name:   string corto (e.g. "evaluate", "tournament", "train")
    config: dict de parametros del run (e.g. vars(args))
    result: dict de metricas devueltas

    Devuelve la ruta absoluta al directorio del run.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(EXPERIMENTS_DIR, f"{ts}_{name}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_jsonify(config), f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(_jsonify(result), f, indent=2, sort_keys=True)

    meta = {
        "name": name,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "git_dirty": _git_dirty(),
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"  Experimento logueado en: experiments/{ts}_{name}/")
    return run_dir
