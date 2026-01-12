INSTALL httpfs;
LOAD httpfs;

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.uber_fares AS
SELECT *
FROM read_parquet(
  'https://huggingface.co/datasets/Carson-Shively/uber-fares/resolve/main/data/bronze/bronze_uf.parquet'
);