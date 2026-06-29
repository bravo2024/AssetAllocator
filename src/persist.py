from __future__ import annotations
import pickle
from pathlib import Path


def save_model(model: object, path: str = "models/model.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "models/model.pkl") -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
