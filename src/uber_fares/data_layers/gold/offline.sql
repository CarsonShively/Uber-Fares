CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE VIEW gold.uber_fares AS
WITH base AS (
  SELECT
    pickup_datetime,
    passenger_count,
    pickup_latitude,  pickup_longitude,
    dropoff_latitude, dropoff_longitude,
    fare_amount
  FROM silver.uber_fares
),
f AS (
  SELECT
    t_year(pickup_datetime)         AS t_year,
    t_month(pickup_datetime)        AS t_month,
    t_day_of_week(pickup_datetime)  AS t_dow,
    t_hour(pickup_datetime)         AS t_hour,
    t_weekend(pickup_datetime)      AS t_weekend,

    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude,

    haversine_mi(pickup_latitude, pickup_longitude,
                 dropoff_latitude, dropoff_longitude) AS geo_haversine_mi,
    manhattan_mi(pickup_latitude, pickup_longitude,
                 dropoff_latitude, dropoff_longitude) AS geo_manhattan_mi,

    bearing_sin(pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude) AS geo_bearing_sin,
    bearing_cos(pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude) AS geo_bearing_cos,

    epsg_sdx_m(pickup_latitude, pickup_longitude,
               dropoff_latitude, dropoff_longitude)  AS epsg_sdx_m,
    epsg_sdy_m(pickup_latitude, pickup_longitude,
               dropoff_latitude, dropoff_longitude)  AS epsg_sdy_m,
    epsg_dx_m(pickup_latitude, pickup_longitude,
              dropoff_latitude, dropoff_longitude)   AS epsg_dx_m,
    epsg_dy_m(pickup_latitude, pickup_longitude,
              dropoff_latitude, dropoff_longitude)   AS epsg_dy_m,
    epsg_manhattan_m(pickup_latitude, pickup_longitude,
                     dropoff_latitude, dropoff_longitude)        AS epsg_manhattan_m,
    epsg_euclid_m(pickup_latitude, pickup_longitude,
                  dropoff_latitude, dropoff_longitude)           AS epsg_euclid_m,


    passenger_count AS passenger_count_clean,
  
    pickup_datetime, passenger_count,
    pickup_latitude, pickup_longitude,
    dropoff_latitude, dropoff_longitude
  FROM base
)

SELECT
  b.fare_amount AS label,
  f.passenger_count_clean,
  f.t_month, f.t_dow, f.t_hour, f.t_weekend,
  f.pickup_latitude, f.pickup_longitude,
  f.dropoff_latitude, f.dropoff_longitude,
  f.geo_haversine_mi, f.geo_manhattan_mi, f.geo_bearing_sin, f.geo_bearing_cos,
  f.epsg_sdx_m, f.epsg_sdy_m, f.epsg_dx_m, f.epsg_dy_m, f.epsg_manhattan_m, f.epsg_euclid_m,
  f.pickup_datetime
FROM f
JOIN base b USING (pickup_datetime, passenger_count,
                   pickup_latitude, pickup_longitude,
                   dropoff_latitude, dropoff_longitude);
