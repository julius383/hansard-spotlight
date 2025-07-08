-- create or replace temp table counties as
select
    cast(COUNTY_NO as utinyint) as county_no,
    COUNTY_NAME as county_name,
    cast(COUNTY_POPULATION as ubigint) as county_population,
    cast(REG_VOTERS as ubigint) as registered_voters,
    AREA_SQ_KM as area_sq_km
from read_json_auto('data/raw/geographical/county_2022.json');
