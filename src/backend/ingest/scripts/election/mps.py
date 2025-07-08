import pandas as pd
import re
import json
from pathlib import Path


def clean_name(s: str) -> str:
    s = re.sub(r'^HON\.\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\(.+\)\s*', '', s, flags=re.IGNORECASE)
    s = ' '.join(map(str.capitalize, s.split(' ')))
    return s

recs = []
for f in Path('data/raw/election/constituency').iterdir():
    with open(f) as fp:
        objs = json.load(fp)
        for o in objs:
            const_no, year = f.stem.split('_')
            o['full_name'] = clean_name(o['candidate'])
            o['vote_count'] = o['votes']
            o['vote_percent'] = o['percentage'].strip('%')
            o['party_short'] = o['party']
            o['constituency_no'] = int(const_no)
            o['year'] = int(year)
            del o['candidate']
            del o['party']
            del o['votes']
            del o['percentage']
            recs.append(o)
mps = pd.DataFrame(recs)
