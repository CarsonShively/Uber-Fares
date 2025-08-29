# feature_store/fs_offline.py
import duckdb, yaml, pandas as pd
from typing import Optional, Sequence, Dict
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # ok if you use a local path

def load_registry(path="feature_store/registry.yaml") -> Dict:
    with open(path) as f:
        reg = yaml.safe_load(f)
    ent = reg["entities"][0]
    src = reg["sources"][0]
    fv  = reg["feature_views"][0]
    return {
        "join_key": ent["join_key"],
        "event_ts": src["event_timestamp"],
        "created_ts": src.get("created_timestamp"),
        "path": src.get("path"),
        "repo_id": src.get("repo_id"),
        "filename": src.get("filename"),
        "revision": src.get("revision", "main"),
        "features": [f["name"] for f in fv["schema"]],
        "label": (reg.get("label") or {}).get("name"),
        "serving_order": reg.get("serving_order"),
    }

def resolve_gold_path(R: Dict) -> str:
    if R.get("path"):                 # local/cached file
        return R["path"]
    if hf_hub_download:               # pull from HF dataset
        return hf_hub_download(
            repo_id=R["repo_id"], repo_type="dataset",
            filename=R["filename"], revision=R["revision"]
        )
    raise RuntimeError("No local path and huggingface_hub not installed.")

def validate_gold(gold_path: str, R: Dict):
    con = duckdb.connect()
    cols = {c[0] for c in con.sql(f"DESCRIBE read_parquet('{gold_path}')").fetchall()}
    required = {R["join_key"], R["event_ts"], *R["features"]}
    if R["label"]: required.add(R["label"])
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"Gold missing columns: {missing}")

def get_historical_features(entity_df: pd.DataFrame,
                            features: Optional[Sequence[str]] = None,
                            include_label: bool = True) -> pd.DataFrame:
    R = load_registry()
    gold = resolve_gold_path(R); validate_gold(gold, R)
    feats = features or R["features"]
    sel_feats = ", ".join([f"g.{c}" for c in feats])
    lbl = f", g.{R['label']}" if include_label and R["label"] else ""
    tie = f", COALESCE(g.{R['created_ts']}, g.{R['event_ts']}) DESC" if R["created_ts"] else ""
    con = duckdb.connect(); con.execute("SET TimeZone='UTC'"); con.register("entities", entity_df)
    q = f"""
    WITH g AS (SELECT * FROM read_parquet('{gold}')),
    ranked AS (
      SELECT
        e.{R['join_key']} AS {R['join_key']},
        e.event_timestamp AS event_timestamp,
        {sel_feats}{lbl},
        ROW_NUMBER() OVER (
          PARTITION BY e.{R['join_key']}, e.event_timestamp
          ORDER BY g.{R['event_ts']} DESC{tie}
        ) rn
      FROM entities e
      JOIN g
        ON g.{R['join_key']} = e.{R['join_key']}
       AND g.{R['event_ts']} <= e.event_timestamp
    )
    SELECT * EXCLUDE (rn) FROM ranked WHERE rn=1
    """
    return con.sql(q).to_df()

def serving_order() -> Sequence[str]:
    R = load_registry()
    return R["serving_order"] or R["features"]
