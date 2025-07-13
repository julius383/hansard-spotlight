#!/usr/bin/env python

import json
import re
from pathlib import Path

from snakemake.script import snakemake
from toolz import compose

from backend.util.helpers import clean_person_name, nullify

functions = {
    "name": clean_person_name,
    "seat": compose(nullify, str.capitalize),
    "party_short": compose(nullify, str.upper),
    "profile_url": nullify,
    "profile_photo": nullify,
    "session_year": nullify,
}


def main():
    # Access input/output files from Snakemake
    input_files = snakemake.input
    output_file = snakemake.output[0]

    data = []
    processed_data = []
    for input_file in input_files:
        if session_year := re.search(r"(20(?:13|17|22))", input_file):
            session_year = int(session_year.group(1))
        else:
            session_year = None

        # Read JSONL file
        with open(input_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                item["session_year"] = session_year
                data.append(item)

    for item in data:
        result = {key: functions[key](item[key]) for key in functions.keys()}
        processed_data.append(result)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            sorted(
                processed_data, key=lambda x: x["session_year"], reverse=True
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
