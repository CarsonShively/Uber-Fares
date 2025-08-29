# loader.py
from pathlib import Path
from typing import Optional
import requests, yaml, duckdb, pandas as pd
from huggingface_hub import hf_hub_download

def load_registry(registry: str) -> dict:
    """Load registry.yaml from a raw GitHub URL or local path."""
    if registry.startswith(("http://", "https://")):
        r = requests.get(registry); r.raise_for_status()
        return yaml.safe_load(r.text)
    with open(registry, "r") as f:
        return yaml.safe_load(f)

def resolve_gold_path(reg: dict) -> str:
    """Return a local path to the HF-cached Parquet (or local override if present)."""
    src = reg["sources"][0]
    local = src.get("path")
    if local and Path(local).exists():
        return local
    return hf_hub_download(
        repo_id=src["repo_id"],
        repo_type="dataset",
        filename=src["filename"],
        revision=src.get("revision", "main"),
    )

def load_gold_df(registry: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read rows from Gold into a DataFrame (optionally limited)."""
    reg = load_registry(registry)
    gold = resolve_gold_path(reg)
    con = duckdb.connect()
    q = f"SELECT * FROM read_parquet('{gold}')" + (f" LIMIT {limit}" if limit else "")
    return con.sql(q).to_df()

# Optional: tiny smoke test when running this file directly
if __name__ == "__main__":
    REG_URL = "https://raw.githubusercontent.com/CarsonShively/Uber-Fares/main/feature-store/registry.yaml"
    df = load_gold_df(REG_URL, limit=5)
    print(df)           # preview only (remove if you truly want zero output)

