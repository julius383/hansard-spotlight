import hashlib
import json
import re
import time
from functools import partial, wraps
from pathlib import Path

from loguru import logger
from toolz import keyfilter, pipe

type AFile = str | Path


def hash_file(file: AFile) -> str:
    with open(file, "rb") as fp:
        digest = hashlib.file_digest(fp, "sha256")
    return digest.hexdigest()


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


def write_results(path: AFile, mode: str = "a"):
    """Decorator that saves output of function to `path` as json"""

    def decorator_save_output(func):
        @wraps(func)
        def wrapper_save_output(*args, **kwargs):
            res = func(*args, **kwargs)
            to = Path(path).expanduser()
            try:
                if to.exists() and mode != "a":
                    new_path = to.with_stem(
                        f"{to.stem}-{int(time.monotonic())}"
                    )
                    logger.info(
                        f"{path} already exists writing to {new_path}",
                        func_name=func.__name__,
                    )
                    to = new_path
                output_file = to

                with open(output_file, mode, encoding="utf-8") as f:
                    json.dump(res, f, default=str, ensure_ascii=False)
                    f.write("\n")
            except TypeError:
                logger.exception("Could not encode json")
            return res

        return wrapper_save_output
    return decorator_save_output
