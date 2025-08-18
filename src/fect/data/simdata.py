from importlib import resources
import pandas as pd


def load() -> pd.DataFrame:
    with resources.files(__package__).joinpath("simdata.csv").open("rb") as f:
        return pd.read_csv(f)

try:
    df = load()
except Exception:
    df = None
