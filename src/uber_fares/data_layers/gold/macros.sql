SET timezone = 'UTC';

INSTALL spatial;
LOAD spatial;

CREATE OR REPLACE MACRO t_year(ts)         AS year(ts);
CREATE OR REPLACE MACRO t_month(ts)        AS month(ts);
CREATE OR REPLACE MACRO t_day_of_week(ts)  AS dayofweek(ts); 
CREATE OR REPLACE MACRO t_hour(ts)         AS hour(ts);
CREATE OR REPLACE MACRO t_weekend(ts)      AS (dayofweek(ts) IN (0,6));

CREATE OR REPLACE MACRO rad(x) AS x * pi() / 180.0;

CREATE OR REPLACE MACRO haversine_mi(lat1, lon1, lat2, lon2) AS (
  3958.761316 * 2 * asin(
    sqrt(
      least(1.0, greatest(0.0,
        sin((rad(lat2 - lat1))/2)^2
        + cos(rad(lat1)) * cos(rad(lat2)) * sin((rad(lon2 - lon1))/2)^2
      ))
    )
  )
);

CREATE OR REPLACE MACRO manhattan_mi(lat1, lon1, lat2, lon2) AS (
  (110.574 * abs(lat2 - lat1)
   + 111.320 * cos(rad((lat1 + lat2)/2.0)) * abs(lon2 - lon1)) * 0.621371
);

CREATE OR REPLACE MACRO _bearing_rad(lat1, lon1, lat2, lon2) AS (
  (atan2(
      sin(rad(lon2 - lon1)) * cos(rad(lat2)),
      cos(rad(lat1)) * sin(rad(lat2))
      - sin(rad(lat1)) * cos(rad(lat2)) * cos(rad(lon2 - lon1))
  ) + 2*pi()) % (2*pi())
);
CREATE OR REPLACE MACRO bearing_sin(lat1, lon1, lat2, lon2) AS sin(_bearing_rad(lat1, lon1, lat2, lon2));
CREATE OR REPLACE MACRO bearing_cos(lat1, lon1, lat2, lon2) AS cos(_bearing_rad(lat1, lon1, lat2, lon2));

CREATE OR REPLACE MACRO _pt_2263(lat, lon) AS
  ST_Transform(
    ST_Point(CAST(lat AS DOUBLE), CAST(lon AS DOUBLE)),
    'EPSG:4326', 'EPSG:2263'
  );
CREATE OR REPLACE MACRO _x_2263_ft(lat, lon) AS ST_X(_pt_2263(lat, lon));
CREATE OR REPLACE MACRO _y_2263_ft(lat, lon) AS ST_Y(_pt_2263(lat, lon));
CREATE OR REPLACE MACRO _ft_to_m(x) AS x * 0.3048;

CREATE OR REPLACE MACRO proj_x_m(lat, lon) AS _ft_to_m(_x_2263_ft(lat, lon));
CREATE OR REPLACE MACRO proj_y_m(lat, lon) AS _ft_to_m(_y_2263_ft(lat, lon));

CREATE OR REPLACE MACRO epsg_sdx_m(lat1, lon1, lat2, lon2) AS proj_x_m(lat2, lon2) - proj_x_m(lat1, lon1);
CREATE OR REPLACE MACRO epsg_sdy_m(lat1, lon1, lat2, lon2) AS proj_y_m(lat2, lon2) - proj_y_m(lat1, lon1);
CREATE OR REPLACE MACRO epsg_dx_m (lat1, lon1, lat2, lon2) AS abs(epsg_sdx_m(lat1, lon1, lat2, lon2));
CREATE OR REPLACE MACRO epsg_dy_m (lat1, lon1, lat2, lon2) AS abs(epsg_sdy_m(lat1, lon1, lat2, lon2));
CREATE OR REPLACE MACRO epsg_manhattan_m(lat1, lon1, lat2, lon2) AS epsg_dx_m(lat1, lon1, lat2, lon2) + epsg_dy_m(lat1, lon1, lat2, lon2);
CREATE OR REPLACE MACRO epsg_euclid_m  (lat1, lon1, lat2, lon2) AS sqrt(epsg_sdx_m(lat1, lon1, lat2, lon2)^2 + epsg_sdy_m(lat1, lon1, lat2, lon2)^2);

