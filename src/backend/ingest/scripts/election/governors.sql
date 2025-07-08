select
    GOVERNOR_ID as id,
    CANDIDATE_VOTES as vote_count,
    cast(trim(COUNTY_NO) as UTINYINT) as county_no,
    YEAR as year,
    round(COUNTY_PERCENTAGE, 4) as vote_percent,
    trim(FIRST_NAME) as first_name,
    trim(LAST_NAME) as last_name,
    FIRST_NAME || ' ' || LAST_NAME as full_name,
    replace(PARTY_INITIALS, ' ', '') as party_short
from read_json_auto('data/raw/election/governor/governor_*', FORMAT = 'array');
