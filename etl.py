import os, pathlib, duckdb
from importlib.resources import files
from huggingface_hub import upload_file


DATASET_REPO_ID = "Carson-Shively/uber-fares"
REVISION        = "main"
EXPORT_REL      = "gold.uber_fares"
LOCAL_FILE      = pathlib.Path("data/gold/gold_uf.parquet")
PATH_IN_REPO    = "data/gold/gold_uf.parquet"

BRONZE_SQL_PKG  = "uber_fares.data_layers.bronze"
SILVER_SQL_PKG  = "uber_fares.data_layers.silver"
GOLD_SQL_PKG    = "uber_fares.data_layers.gold"

BRONZE_SQL_FILE = "bronze.sql"
SILVER_SQL_FILE = "silver.sql"
MACROS_SQL_FILE = "macros.sql"
OFFLINE_SQL_FILE = "offline.sql"

def _read_pkg_sql(pkg: str, filename: str) -> str:
    return (files(pkg) / filename).read_text(encoding="utf-8")

def _exec_sql_text(con, sql: str):
    con.execute(sql)

def build_and_write_parquet():
    LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        _exec_sql_text(con, _read_pkg_sql(GOLD_SQL_PKG, MACROS_SQL_FILE))
        _exec_sql_text(con, _read_pkg_sql(BRONZE_SQL_PKG, BRONZE_SQL_FILE))
        _exec_sql_text(con, _read_pkg_sql(SILVER_SQL_PKG, SILVER_SQL_FILE))
        _exec_sql_text(con, _read_pkg_sql(GOLD_SQL_PKG, OFFLINE_SQL_FILE))

        con.execute(
            f"COPY (SELECT * FROM {EXPORT_REL}) TO '{LOCAL_FILE.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD);"
        )
    finally:
        con.close()

def upload_parquet():
    info = upload_file(
        path_or_fileobj=str(LOCAL_FILE),
        path_in_repo=PATH_IN_REPO,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        commit_message=f"Upload {PATH_IN_REPO}",
    )
    print("Uploaded", PATH_IN_REPO, getattr(info, "commit_id", ""))

if __name__ == "__main__":
    build_and_write_parquet()
    upload_parquet()
