import pandas as pd
import re
import json
from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values(".env")
data_file = Path(config["APP_ROOT"]) / "data/entities/parliament/files/national_assembly/mps_profiles_2022.jsonl"

def clean_name(s: str) -> str:
    s = re.sub(r'^HON\.\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\(.+\)\s*', '', s, flags=re.IGNORECASE)
    s = ' '.join(map(str.capitalize, s.split(' ')))
    return s


with open(data_file) as fp:
    recs = []
    for line in fp:
        o = json.loads(line)
        o['name'] = clean_name(o['name'])
        o['seat'] = str.capitalize(o['seat'])
        del o['education_history']
        del o['employment_history']
        del o['type']
        recs.append(o)

mps = pd.DataFrame(recs)
