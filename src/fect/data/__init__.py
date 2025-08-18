from __future__ import annotations

from importlib import resources
import pandas as pd

__all__ = [
    "gs2020",
    "hh2019",
    "turnout",
    "simdata",
    "simgsynth",
]

def _load_csv(name: str) -> pd.DataFrame:
    with resources.files(__package__).joinpath(f"{name}.csv").open("rb") as f:
        return pd.read_csv(f)

def gs2020() -> pd.DataFrame:
    return _load_csv("gs2020")

def hh2019() -> pd.DataFrame:
    return _load_csv("hh2019")

def turnout() -> pd.DataFrame:
    return _load_csv("turnout")

def simdata() -> pd.DataFrame:
    return _load_csv("simdata")

def simgsynth() -> pd.DataFrame:
    return _load_csv("simgsynth")
