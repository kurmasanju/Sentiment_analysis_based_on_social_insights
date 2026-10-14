from pathlib import Path
import pandas as pd


def load_source(path: str | Path, source_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    df = pd.read_csv(path)
    df["source"] = source_name
    return df
