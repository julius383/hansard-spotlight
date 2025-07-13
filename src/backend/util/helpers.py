import re
from functools import partial

from toolz import keyfilter, pipe


def pick(allowlist, d):
    return keyfilter(lambda k: k in allowlist, d)


def omit(denylist, d):
    return keyfilter(lambda k: k not in denylist, d)


def reorder_names(s: str) -> str:
    if "," in s:
        names = reversed(s.split(","))
        return " ".join(names)
    return s


def clean_person_name(s: str) -> str:
    s = pipe(
        s,
        str.lower,
        partial(re.sub, r"^hon\.?\s*", ""),
        partial(re.sub, r"^\(.+\)\s*", ""),
        partial(re.sub, r"\s{2,}", " "),
        reorder_names,
        str.title,
        str.strip,
    )
    return s

def nullify(s, f=lambda x: x):
    return None if (isinstance(s, str) and s == "") else f(s)
