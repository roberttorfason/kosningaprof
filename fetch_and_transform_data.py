import argparse
import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from boltons.strutils import html2text


def clean_question_str(s: str) -> str:
    """Removes html from string and truncates the string from the left until first capitalized letter

    NOTE: Just a simple heuristic that happens to work for this set of strings

    >>> clean_question_str("3/31&nbsp;<br> Fjölga þarf lóðum í sveitarfélaginu undir íbúðarhúsnæði.")
    'Fjölga þarf lóðum í sveitarfélaginu undir íbúðarhúsnæði.'
    """
    s_out = html2text(s)
    uppercase_idxs = [i for i, c in enumerate(s_out) if c.isupper()]
    assert uppercase_idxs, "Expected to find at least one capitalized letter for this heuristic"
    idx = uppercase_idxs[0]

    return s_out[idx:].strip()


Question = namedtuple("QuestionRow", "id, question")


def process_questions(questions: dict) -> pd.DataFrame:
    rows = [Question(id=str(q["id"]), question=q["title"]) for i, q in enumerate(questions)]
    df_questions = pd.DataFrame(rows)
    df_questions["question"] = df_questions["question"].map(clean_question_str)
    return df_questions


def load_data(p: Union[Path, str]) -> dict:
    with open(p, "r") as f:
        d = json.load(f)
    return d


def _extract_digits(_txt: str) -> int:
    ints = [int(s) for s in _txt.split() if s.isdigit()]
    if ints:
        return ints[0]
    else:
        return -1


FieldToExtract = namedtuple("Info", "name, default")


def extract_target(user_info: dict, fields: List[FieldToExtract]) -> Dict[str, str]:
    target_data = user_info["target_data"]
    return {key: target_data.get(key, default) for key, default in fields}


def process_results(results: dict) -> pd.DataFrame:
    rows = []
    for result in results:
        if "target_data" in result:
            d = {
                **extract_target(
                    result, [FieldToExtract("age", "-1"), FieldToExtract("party", ""), FieldToExtract("gender", "")]
                ),
                **result["target_values"],
                "name": result["title"],
                "answering_done": result.get("answering_done", 0.0),
            }
            rows.append(d)
    df = pd.DataFrame(rows)
    df["email"] = df["email"].fillna("")
    df["age"] = df["age"].map(_extract_digits)
    df["answering_done"] = df["answering_done"].view(float)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["party"] = df["party"].astype("category")
    df["gender"] = df["gender"].astype("category")

    return df


def move_cols_to_front(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df


CONSTITUENCY_ID = "4768672638828544"
BASEURL = "https://apps.myzef.com/resources/ruv/jqkdxx/6115134333255680/"
QUESTION_FNAME = "questions-6415658043572224.js"
RESULTS_FNAME = "results-5660526221721600.js"


def fetch_data(f_name: str) -> dict:
    r = requests.get(BASEURL + f_name, allow_redirects=True)
    return json.loads(r.text)


def main(p: Optional[Path], output_path: Path, keep_incomplete: bool):
    os.makedirs(output_path, exist_ok=True)

    if p is None:
        questions = fetch_data(QUESTION_FNAME)
        results = fetch_data(RESULTS_FNAME)

        with open(output_path / QUESTION_FNAME, "w") as f:
            json.dump(questions, f, ensure_ascii=False)
        with open(output_path / RESULTS_FNAME, "w") as f:
            json.dump(results, f, ensure_ascii=False)
    else:
        questions = load_data(p / QUESTION_FNAME)
        results = load_data(p / RESULTS_FNAME)

    questions, results = questions["children"], results["children"]

    df_questions = process_questions(questions)
    df_questions = df_questions[df_questions["id"] != CONSTITUENCY_ID]
    df_questions = df_questions.reset_index(drop=True)
    df_questions.index.name = "question_number"
    df_questions = df_questions.reset_index(drop=False)
    df_questions["question_number"] = "question_" + df_questions["question_number"].astype(str)
    df_questions.to_csv(output_path / "questions_2021.csv", index=False)

    encoding_to_constituency = {d["value"]: d["label"] for d in questions[0]["data"]["choices"]}

    df_results = process_results(results)
    if not keep_incomplete:
        df_results = df_results[df_results["answering_done"] == 1.0]
    question_cols_idx = df_results.columns.isin(df_questions["id"].astype(str))
    cols_meta = df_results.columns[~question_cols_idx]
    df_results = move_cols_to_front(df_results, cols_meta.to_list())
    df_results = move_cols_to_front(df_results, ["name", "party"])
    df_results = df_results.rename(columns={CONSTITUENCY_ID: "constituency"})
    df_results["constituency"] = df_results["constituency"].map(encoding_to_constituency).fillna("")
    df_results = df_results.drop(columns="email")

    question_id_to_number = dict(zip(df_questions["id"], df_questions["question_number"]))
    # TODO: df['col1'].map(di).fillna(df['col1'])
    df_results.columns = df_results.columns.map(lambda c: question_id_to_number[c] if c in question_id_to_number else c)

    df_results.to_csv(output_path / "results_2021.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", type=str, help="If not specified will download the data. If specified will load data locally from this folder.")
    parser.add_argument("--output-path", type=str, default="data", help="Where the files will be saved")
    parser.add_argument("--keep-incomplete", action="store_true", help="By default candidates that have not finished answering all questions are discarded. If this flag is used, they will not be discarded.")
    args = parser.parse_args()

    main(None if args.local_path is None else Path(args.local_path), Path(args.output_path), args.keep_incomplete)
