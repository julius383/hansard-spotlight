#!/usr/bin/env python

import re
import json
from pathlib import Path
from functools import partial

from snakemake.script import snakemake
from toolz import compose, identity

from backend.util.helpers import clean_person_name, nullify


functions = {
    "candidate": clean_person_name,
    "party": compose(nullify, str.upper),
    "votes": lambda x: nullify(x, int),
    "percentage": compose(lambda x: nullify(x, float), partial(re.sub, r'%', '')),
    "constituency_no": identity,
    "year": identity,
}


def main():
    input_files = snakemake.input
    output_file = snakemake.output[0]

    data = []
    processed_data = []
    for input_file in input_files:
        if m := re.search(r"(\d{3})_(20(?:13|17|22))", input_file):
            constituency_no = int(m.group(1))
            year = int(m.group(2))
        else:
            constituency_no = None
            year = None

        with open(input_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                item["constituency_no"] = constituency_no
                item["year"] = year
                data.append(item)

    for item in data:
        result = {key: functions[key](item[key]) for key in functions.keys()}
        processed_data.append(result)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            sorted(
                processed_data, key=lambda x: x["year"], reverse=True
            ),
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
