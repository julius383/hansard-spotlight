import json
import re
import uuid
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Dict, Optional, Union

import duckdb
from rapidfuzz import fuzz, process, utils
from toolz import compose

# DATA_DIR = Path("../../../") / "data/results/"
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

    # Default weights - can be tuned based on your data characteristics
    if weights is None:
        weights = {
            "ratio": 0.15,  # Basic Levenshtein ratio
            "token_sort_ratio": 0.35,  # Sorted token comparison
            "token_set_ratio": 0.35,  # Set-based token comparison
            "qratio": 0.15,  # Quick ratio (fastest)
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

    # Calculate individual scores
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

    # Calculate weighted combination
    combined_score = sum(scores[method] * weights[method] for method in weights)

    if return_components:
        scores["combined_score"] = combined_score
        return scores

    return combined_score


def main():
    # link profiles and elections
    profiles = duckdb.read_json(str(DATA_DIR / "parliament_profiles.json"))
    election_data = duckdb.read_json(str(DATA_DIR / "election_mps.json"))
    constituencies = duckdb.read_json(str(DATA_DIR / "constituency.json"))
    committees = duckdb.read_json(str(DATA_DIR / "committees.json"))

    # disambiguate constituency
    profiles_with_constituency = duckdb.sql(
        """
        select prof.*, constituency_no
        from profiles prof
        inner join constituencies const
        on lcase(prof.seat) = lcase(const.constituency_name)
        """
    )
    name_mapping = {}
    for profile in duckdb.sql(
        "select * from profiles_with_constituency where session_year = 2022"
    ).fetchall():
        name, seat, party, profile_url, profile_photo, year, constituency_no = (
            profile
        )
        akas = [{"dataset": "parliament_profiles.json", "id": name}]

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
                {"dataset": "election_mps.json", "id": election_name[0]}
            )

        if committee_name:
            akas.append({"dataset": "committees.json", "id": committee_name[0]})

        id_ = uuid.uuid4()
        name_mapping[str(id_)] = akas
    with open(DATA_DIR / "linkage.json", "w") as fp:
        json.dump(name_mapping, fp, indent=2)
    return name_mapping
