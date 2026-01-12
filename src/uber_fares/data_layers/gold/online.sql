CREATE OR REPLACE MACRO online_q_all(
  pickup_datetime TIMESTAMPTZ,
  passenger_count DOUBLE,
  pickup_latitude DOUBLE,
  pickup_longitude DOUBLE,
  dropoff_latitude DOUBLE,
  dropoff_longitude DOUBLE
) AS TABLE (
  SELECT
    t_month(pickup_datetime)       AS t_month,
    t_day_of_week(pickup_datetime) AS t_dow,
    t_hour(pickup_datetime)        AS t_hour,
    t_weekend(pickup_datetime)     AS t_weekend,

    CAST(greatest(0, least(8, round(passenger_count))) AS INT) AS passenger_count_clean,
    (pickup_latitude)   AS pickup_latitude,
    (pickup_longitude)  AS pickup_longitude,
    (dropoff_latitude)  AS dropoff_latitude,
    (dropoff_longitude) AS dropoff_longitude,

    haversine_mi(pickup_latitude, pickup_longitude,
                 dropoff_latitude, dropoff_longitude) AS geo_haversine_mi,
    manhattan_mi(pickup_latitude, pickup_longitude,
                 dropoff_latitude, dropoff_longitude) AS geo_manhattan_mi,
    bearing_sin(pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude)  AS geo_bearing_sin,
    bearing_cos(pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude)  AS geo_bearing_cos,
    epsg_sdx_m(pickup_latitude, pickup_longitude,
               dropoff_latitude, dropoff_longitude)   AS epsg_sdx_m,
    epsg_sdy_m(pickup_latitude, pickup_longitude,
               dropoff_latitude, dropoff_longitude)   AS epsg_sdy_m,
    epsg_dx_m(pickup_latitude, pickup_longitude,
              dropoff_latitude, dropoff_longitude)    AS epsg_dx_m,
    epsg_dy_m(pickup_latitude, pickup_longitude,
              dropoff_latitude, dropoff_longitude)    AS epsg_dy_m,
    epsg_manhattan_m(pickup_latitude, pickup_longitude,
                     dropoff_latitude, dropoff_longitude) AS epsg_manhattan_m,
    epsg_euclid_m(pickup_latitude, pickup_longitude,
                  dropoff_latitude, dropoff_longitude)    AS epsg_euclid_m
);

