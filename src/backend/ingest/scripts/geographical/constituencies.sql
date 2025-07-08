-- create or replace temp table constituencies as
select
    cast(COUNTY_NO as utinyint) as county_no,
    cast(CONSTITUENCY_NO as uinteger) as constituency_no,
    cast(CONSTITUENCY_POPULATION as ubigint) as county_population,
    CONSTITUENCY_NAME as constituency_name,
    cast(REG_VOTERS as ubigint) as registered_voters,
    AREA_SQ_KM as area_sq_km
from read_json_auto('data/raw/geographical/constituency_2022.json')
where cast(CONSTITUENCY_NO as uinteger) <= 290;
