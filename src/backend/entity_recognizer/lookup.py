from pathlib import Path

import duckdb
from duckdb.typing import VARCHAR, FLOAT    # noqa: F401

from backend.util.helpers import clean_person_name

DATA_DIR = Path("data/results")


# FIXME: improve lookup, possibly with record linkage
def lookup_name(name: str) -> str:
    links = duckdb.read_json(DATA_DIR / "linkage.json")     # noqa: F841
    name = clean_person_name(name)
    # TODO: investigate using different weighting for lookup

    # duckdb.create_function("fuzzy_score", lambda x: combined_fuzzy_score(x, name), [VARCHAR], FLOAT)
    results = duckdb.default_connection().execute(
        "select * from links where gid = (select gid from links where lower(id) = ?)",
        [name],
    )
    context = []
    for item in results.fetchall():
        dataset, id, gid, id_col = item
        data = duckdb.read_json(DATA_DIR / dataset)
        ctx = None
        match dataset:
            case "election_mps.json":
                ctx = duckdb.default_connection().execute(
                    "select * from data where ? = ?", [id_col, id]
                ).fetchdf().to_dict("records")
            case "parliament_profiles.json":
                ctx = duckdb.default_connection().execute(
                    "select * from data where ? = ?", [id_col, id]
                ).fetchdf().to_dict("records")
            case "committees.json":
                ctx = duckdb.default_connection().execute(
                    "select * from data where ? in ?", [id, id_col]
                ).fetchdf().to_dict("records")
        if ctx is not None:
            context.append(ctx)
    return context
