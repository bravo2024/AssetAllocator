from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def save_metrics(metrics: dict, path: str = "models/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            cleaned[k] = int(v)
        elif isinstance(v, dict):
            cleaned[k] = {
                sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                for sk, sv in v.items()
            }
        else:
            cleaned[k] = v
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)
