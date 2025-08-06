import json
import re
from functools import partial
from typing import Any, Dict, List

from toolz import pipe

from backend.util.helpers import clean_person_name


def split_on(
    xs: [str],
    rx=r"^[A-Z]{1,2}\..+",
) -> [[str]]:
    res = []
    group = []
    for line in xs:
        if re.match(rx, line):
            if group:
                res.append(group)
                group = []
        group.append(line)
    return res


def _clean_name(s):
    return pipe(
        s,
        partial(re.sub, r"^\d+\. The \w+\.\s*", ""),
        partial(re.sub, r",?\s*([A-Z]{3})?,\s*M\.?P\.?$", ""),
        partial(re.sub, r"\n", " "),
        clean_person_name,
    )


def _process(item):
    name = pipe(
        item[0],
        partial(re.sub, r"^[A-Z]{1,2}\.\s*", ""),
        str.title,
        str.strip,
    )
    chair = _clean_name(item[1])
    vice = _clean_name(item[2])
    d = {
        "committee_name": name,
        "chair": chair,
        "vice": vice,
        "members": [cleaned for i in item[1:] if (cleaned := _clean_name(i))],
    }
    return d


def main():
    with open("../../../data/entities/parliament/committees.txt", "r") as fp:
        txt = fp.readlines()
    committees = split_on(txt)
    processed = []
    for item in committees:
        name = pipe(
            item[0],
            partial(re.sub, r"^[A-Z]{1,2}\.\s*", ""),
            str.capitalize,
            str.strip,
        )
        chair = _clean_name(item[1])
        vice = _clean_name(item[2])
        d = {
            "committee_name": name,
            "chair": chair,
            "vice": vice,
            "members": [_clean_name(i) for i in item[1:]],
        }
        processed.append(d)

    with open("./committees.json", 'w') as fp:
        json.dump(processed, fp, indent=2)


if __name__ == "__main__":
    main()
