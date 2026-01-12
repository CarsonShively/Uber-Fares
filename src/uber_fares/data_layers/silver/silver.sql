CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE VIEW silver.uber_fares AS
WITH base AS (
  SELECT * FROM bronze.uber_fares
),
  
typed AS (
  SELECT
    CAST("key" AS VARCHAR)                AS trip_key,
    TRY_CAST(fare_amount        AS DOUBLE)      AS fare_amount,
    TRY_CAST(passenger_count    AS INTEGER)     AS passenger_count,
    TRY_CAST(pickup_datetime    AS TIMESTAMPTZ) AS pickup_datetime, 
    TRY_CAST(pickup_longitude   AS DOUBLE)      AS pickup_longitude,
    TRY_CAST(pickup_latitude    AS DOUBLE)      AS pickup_latitude,
    TRY_CAST(dropoff_longitude  AS DOUBLE)      AS dropoff_longitude,
    TRY_CAST(dropoff_latitude   AS DOUBLE)      AS dropoff_latitude
  FROM base
),

valid AS (
  SELECT
    trip_key,
    pickup_datetime,
    CASE WHEN fare_amount > 0 AND fare_amount < 500 THEN fare_amount END AS fare_amount,
    CASE WHEN passenger_count BETWEEN 1 AND 6 THEN passenger_count END   AS passenger_count,
    CASE WHEN pickup_longitude BETWEEN -75 AND -72 THEN pickup_longitude END   AS pickup_longitude,
    CASE WHEN pickup_latitude  BETWEEN  40 AND  42 THEN pickup_latitude  END   AS pickup_latitude,
    CASE WHEN dropoff_longitude BETWEEN -75 AND -72 THEN dropoff_longitude END AS dropoff_longitude,
    CASE WHEN dropoff_latitude  BETWEEN  40 AND  42 THEN dropoff_latitude  END AS dropoff_latitude
  FROM typed
),

norm AS (
  SELECT
    DATE_TRUNC('minute', pickup_datetime) AS ts_min,
    ROUND(pickup_longitude, 5)  AS p_lon5,
    ROUND(pickup_latitude, 5)   AS p_lat5,
    ROUND(dropoff_longitude, 5) AS d_lon5,
    ROUND(dropoff_latitude, 5)  AS d_lat5,
    *
  FROM valid
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY ts_min, p_lon5, p_lat5, d_lon5, d_lat5, passenger_count
      ORDER BY
        (fare_amount IS NULL),
        (pickup_datetime IS NULL),
        (pickup_longitude IS NULL) OR NOT isfinite(pickup_longitude),
        (pickup_latitude  IS NULL) OR NOT isfinite(pickup_latitude),
        (dropoff_longitude IS NULL) OR NOT isfinite(dropoff_longitude),
        (dropoff_latitude  IS NULL) OR NOT isfinite(dropoff_latitude),
        (trip_key IS NULL),
        pickup_datetime ASC, 
        trip_key ASC
    ) AS rn
  FROM norm
)
SELECT * EXCLUDE (rn, ts_min, p_lon5, p_lat5, d_lon5, d_lat5)
FROM ranked
WHERE rn = 1
  AND trip_key IS NOT NULL
  AND pickup_datetime IS NOT NULL
  AND fare_amount IS NOT NULL
  AND passenger_count IS NOT NULL
  AND pickup_longitude IS NOT NULL
  AND pickup_latitude  IS NOT NULL
  AND dropoff_longitude IS NOT NULL
  AND dropoff_latitude  IS NOT NULL
  AND isfinite(fare_amount)
  AND isfinite(pickup_longitude)
  AND isfinite(pickup_latitude)
  AND isfinite(dropoff_longitude)
  AND isfinite(dropoff_latitude);