import pandas as pd
import re
import json
from pathlib import Path
from toolz import keyfilter, keymap, valmap, identity


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def clean_name(s: str) -> str:
    s = re.sub(r"^HON\.\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\(.+\)\s*", "", s, flags=re.IGNORECASE)
    s = " ".join(map(str.capitalize, s.split(" ")))
    return s


recs = []
relevant = [
    "president_id",
    "first_name",
    "last_name",
    "party_initials",
    "candidate_votes",
    "county_percentage",
    "county_no",
]
for f in Path("data/raw/election/president").iterdir():
    with open(f) as fp:
        obj = json.load(fp)
        national = obj["county"]
        for c in national["candidates"]:
            procd = pick(relevant, keymap(str.lower, c))
            procd = valmap(
                lambda x: str.strip(x) if isinstance(x, str) else identity(x),
                procd
            )
            if procd["candidate_votes"] is None or procd["candidate_votes"] == 0:
                continue
            procd["candidate_votes"] = int(procd["candidate_votes"])
            procd["county_no"] = int(procd["county_no"])
            procd["percentage"] = round(procd["county_percentage"], 2)
            del procd["county_percentage"]
            procd["year"] = int(re.search(r'president_(\d{4}).json$', str(f)).group(1))
            procd["full_name"] = procd["first_name"] + ' ' + procd["last_name"]
            recs.append(procd)
county = pd.DataFrame(recs)
