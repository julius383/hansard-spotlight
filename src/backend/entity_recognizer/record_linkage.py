import json
import re
import uuid
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Dict, Optional, Union
from itertools import combinations

import duckdb
from rapidfuzz import fuzz, process, utils
from toolz import compose, groupby, unique
from tqdm import tqdm

DATA_DIR = Path("data/results")


def combined_fuzzy_score(
    str1: str,
    str2: str,
    weights: Optional[Dict[str, float]] = None,
    return_components: bool = False,
    score_cutoff=None,
) -> Union[float, Dict[str, float]]:
    """
    Calculate a weighted combination of rapidfuzz scoring methods for record linkage.

    Parameters:
    -----------
    str1, str2 : str
        Strings to compare
    weights : dict, optional
        Weights for each scoring method. Default: equal weights
        Keys: 'ratio', 'token_sort_ratio', 'token_set_ratio', 'qratio'
    return_components : bool, default False
        If True, returns individual scores along with combined score

    Returns:
    --------
    float or dict
        Combined weighted score (0-100) or dict with all scores if return_components=True
    """

    if weights is None:
        weights = {
            "ratio": 0.15,
            "token_sort_ratio": 0.35,
            "token_set_ratio": 0.35,
            "qratio": 0.15,
        }

    # Handle None/empty strings
    if not str1 or not str2:
        result = 0.0
        if return_components:
            return {
                "ratio": 0.0,
                "token_sort_ratio": 0.0,
                "token_set_ratio": 0.0,
                "qratio": 0.0,
                "combined_score": result,
            }
        return result

    scores = {
        "ratio": fuzz.ratio(str1, str2, score_cutoff=score_cutoff),
        "token_sort_ratio": fuzz.token_sort_ratio(
            str1, str2, score_cutoff=score_cutoff
        ),
        "token_set_ratio": fuzz.token_set_ratio(
            str1, str2, score_cutoff=score_cutoff
        ),
        "qratio": fuzz.QRatio(str1, str2, score_cutoff=score_cutoff),
    }

    combined_score = sum(scores[method] * weights[method] for method in weights)

    if return_components:
        scores["combined_score"] = combined_score
        return scores

    return combined_score


# TODO: backup old linkage results before writing new ones
def main():
    # link profiles and elections
    # ruff: noqa: F841

    profiles = duckdb.read_json(str(DATA_DIR / "parliament_profiles.json"))
    election_data = duckdb.read_json(str(DATA_DIR / "election_mps.json"))
    constituencies = duckdb.read_json(str(DATA_DIR / "constituency.json"))
    committees = duckdb.read_json(str(DATA_DIR / "committees.json"))

    # disambiguate constituency
    profiles_with_constituency = duckdb.sql(
        """
        select prof.name, prof.session_year, constituency_no
        from profiles prof
        inner join constituencies const
        on lcase(prof.seat) = lcase(const.constituency_name)
        """
    )
    name_mapping = []
    for profile in tqdm(
        duckdb.sql(
            "select * from profiles_with_constituency where session_year = 2022"
        ).fetchall()
    ):
        id_ = uuid.uuid4()
        name, year, constituency_no = profile
        akas = [
            {
                "dataset": "parliament_profiles.json",
                "id": name,
                "gid": str(id_),
                "id_col": "name",
            }
        ]

        election_candidates = list(
            map(
                itemgetter(0),
                duckdb.default_connection()
                .execute(
                    """select candidate from election_data where constituency_no = ? and year = ?""",
                    [constituency_no, year],
                )
                .fetchall(),
            )
        )
        election_name = process.extractOne(
            name,
            election_candidates,
            scorer=combined_fuzzy_score,
            score_cutoff=70,
            processor=compose(
                utils.default_process, partial(re.sub, r"&#39;", "'")
            ),
        )

        committee_candidates = list(
            map(
                itemgetter(0),
                duckdb.sql(
                    """
                select distinct member
                from (select unnest(members) as member from committees)
                """
                ).fetchall(),
            )
        )
        committee_name = process.extractOne(
            name,
            committee_candidates,
            scorer=combined_fuzzy_score,
            score_cutoff=70,
            processor=compose(
                utils.default_process, partial(re.sub, r"&#39;", "'")
            ),
        )
        if election_name:
            akas.append(
                {
                    "dataset": "election_mps.json",
                    "id": election_name[0],
                    "gid": str(id_),
                    "id_col": "candidate",
                }
            )

        if committee_name:
            akas.append(
                {
                    "dataset": "committees.json",
                    "id": committee_name[0],
                    "gid": str(id_),
                    "id_col": "members",
                }
            )

        # if len(akas) > 1:
        name_mapping.extend(akas)
    with open(DATA_DIR / "linkage.json", "w") as fp:
        json.dump(name_mapping, fp, indent=2)
    return name_mapping


def lint_linkage(linkage_file: str | Path):
    with open(linkage_file, "r") as fp:
        items = json.load(fp)

    # similar names not fully linked
    names = sorted(unique(map(itemgetter('id'), items)))
    cross = combinations(names, 2)
    res = {}
    for (k1, k2) in cross:
        r = fuzz.QRatio(k1, k2)
        if r >= 70:
            k1_gid = next(i for i in items if i["id"] == k1)
            k2_gid = next(i for i in items if i["id"] == k2)
            if k1_gid["gid"] != k2_gid["gid"]:
                res[(k1, k2)] = ((k1_gid, k2_gid), r)
    return res

if __name__ == "__main__":
    main()
