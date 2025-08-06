import re
from functools import partial
from pathlib import Path
from typing import Any

import pymupdf
import roman
from loguru import logger
from toolz import pipe

from backend.document_processor.headers import extract_headers
from backend.util.helpers import AFile, hash_file, write_results

DATA_DIR = Path("data/results")


@write_results(DATA_DIR / "hansard_metadata.jsonl", mode="a")
def extract_metadata(file: AFile, add_toc=True) -> dict[str, Any]:
    """Extract metadata from Hansard files."""
    with pymupdf.open(file) as doc:
        doc = pymupdf.open(file)
        cover = doc[0].get_text()
    meta = {
        "filename": Path(file).name,
        "sha256": hash_file(file),
        "page_count": doc.page_count,
    }
    if add_toc:
        meta |= {
            "toc": [i.text for i in extract_headers(file) if not i.text.startswith('(')],
        }
    res = pipe(
        cover,
        partial(re.sub, r"(\n *){2,}", "\n"),
        partial(re.sub, r"\n{2,} *", "\n"),
        str.strip,
        partial(re.split, r"\n *"),
        partial(map, str.strip),
        partial(
            filter,
            lambda x: re.match(
                r"(?:.* parliament$)|(?:^vol)", x, re.IGNORECASE
            ),
        ),
        list,
    )
    if len(res) < 2:
        logger.error(f"Failed to parse metadata from {file}")
    else:
        session, id_ = res
        m = re.match(r"vol\. (.+) no\. (.+)", id_, re.IGNORECASE)
        meta |= {
            "session": session,
            "volume": roman.fromRoman(m.group(1)),
            "number": int(m.group(2)),
        }
    return meta

