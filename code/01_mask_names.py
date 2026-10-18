#!/usr/bin/env python3
"""
Replace party and politician names with [PARTY] / [POLITICIAN].

Reads data/<task>.csv and writes data/<task>_masked.csv with the same rows in
the same order, the text columns masked, and a masked_changed flag. Items whose
text has no name are word-for-word identical in both files; they are the
placebo in the analysis.

- party names -> "[PARTY]" (Labour, New Labour, Conservative(s), Tory/Tories,
  Liberal Democrat(s), Lib Dem(s), capitalised Liberal(s), the 1987 Alliance,
  SDP, Social Democrats; Democrat(s)/Democratic Party, Dem(s), Republican(s),
  GOP, MAGA in tweets)
- named politicians -> "[POLITICIAN]" (full names, "Mr/Mrs" + surname, and
  bare surnames that are unambiguous in these texts; politician and party
  Twitter handles; hashtags containing a name, e.g. #BidenFlation)

Kept as content: ideology words in lower case ("conservative values"), the
government/opposition role, slogans and policy terms.

Same rules as code/build_masked_position_tasks.py in polsci-open-bench, which
built the files used in the runs; with --check, the script verifies that it
reproduces them exactly instead of overwriting them.
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TASKS = {
    "benoit_manifesto_economic_position": ["context_before", "text", "context_after"],
    "benoit_manifesto_social_position": ["context_before", "text", "context_after"],
    "lemens_gallego_tweet_position": ["text"],
}

UK_POLITICIANS = [
    r"(?:Mrs\.? |Margaret )Thatcher", r"Thatcher", r"(?:Mr\.? |John )Major", r"Major(?= (?:[Gg]overnments?|years))",
    r"(?:Mr\.? |Gordon )Brown", r"(?:Mr\.? |Tony )?Blair", r"(?:Mr\.? |Michael )Foot(?! and Mouth)",
    r"(?:Mr\.? |David )?Cameron", r"(?:Mr\.? |Neil )?Kinnock", r"(?:Mr\.? |Michael )Howard",
    r"(?:Mr\.? |David )Steel", r"(?:Mr\.? |David )Owen", r"(?:Mr\.? |Paddy )?Ashdown",
    r"(?:Mr\.? |Charles )Kennedy", r"(?:Mr\.? |Nick )?Clegg", r"(?:Mr\.? |William )?Hague",
    r"(?:Mr\.? |Iain )?Duncan Smith", r"(?:Mr\.? |John )Smith",
]
UK_PARTIES = [
    r"New Labour", r"Labour", r"Conservatives?", r"Tory", r"Tories",
    r"Liberal Democrats?", r"Lib Dems?", r"Liberals?(?= [Pp]art)", r"Liberals", r"Liberal(?= (?:insistence|devolution|policies))",
    r"(?<=\[PARTY\] and )Liberal",
    r"SDP", r"Social Democrats?", r"(?<![Aa]tlantic )Alliance",
]
US_POLITICIANS = [
    r"@\w*(?:Biden|Trump|McCarthy|Pelosi|Schumer|McConnell|Jeffries|Harris|Obama|DeSantis|POTUS|VP)\w*",
    r"#\w*(?:Biden|Trump|Pelosi|Schumer|McConnell|Obama|DeSantis)\w*",
    r"(?:President |Joe |Donald |Speaker |Kevin |Nancy |Leader )?"
    r"(?:Biden|BIDEN|Trump|McCarthy|Pelosi|Schumer|McConnell|Jeffries|Obama|DeSantis|Desantis)(?:nomics|'s)?",
    r"(?:Vice President |VP |Kamala )Harris",
]
US_PARTIES = [
    r"@\w*(?:GOP|Democrats|Dems|Republicans)\w*", r"#\w*(?:GOP|MAGA|Democrats|Republicans)\w*",
    r"Democratic Party", r"Republican Party", r"Democratic(?= Issues Conference| Caucus)",
    r"Republicanos", r"Democrats?", r"Dems?", r"Republicans?",
    r"republicans", r"GOP", r"MAGA",
]


def compile_rules(politicians, parties):
    rules = [(re.compile(r"(?<![\w@#])(?:%s)(?!\w)" % p), "[POLITICIAN]") for p in politicians]
    rules += [(re.compile(r"(?<![\w@#])(?:%s)(?!\w)" % p), "[PARTY]") for p in parties]
    # Handles and hashtags start with @ or #, so they are matched without the
    # leading word-boundary guard.
    rules = [(re.compile(r"(?:%s)" % p), "[POLITICIAN]") for p in politicians if p.startswith(("@", "#"))] + rules
    rules = [(re.compile(r"(?:%s)" % p), "[PARTY]") for p in parties if p.startswith(("@", "#"))] + rules
    return rules


def mask(text, rules):
    if not isinstance(text, str):
        return text
    for pattern, placeholder in rules:
        text = pattern.sub(placeholder, text)
    return text


def main():
    check = "--check" in sys.argv
    for task, columns in TASKS.items():
        rules = compile_rules(*(
            (UK_POLITICIANS, UK_PARTIES) if task.startswith("benoit") else (US_POLITICIANS, US_PARTIES)))
        data = pd.read_csv(REPO / "data" / f"{task}.csv")
        masked = data.copy()
        for col in columns:
            masked[col] = data[col].map(lambda t: mask(t, rules))
        masked["masked_changed"] = (masked[columns].fillna("") != data[columns].fillna("")).any(axis=1)
        path = REPO / "data" / f"{task}_masked.csv"
        if check:
            pd.testing.assert_frame_equal(masked, pd.read_csv(path))
            print(f"{task}: reproduces {path.name}")
        else:
            masked.to_csv(path, index=False)
        print(f"{task}: {int(masked.masked_changed.sum())} of {len(data)} rows contain a name")


if __name__ == "__main__":
    main()
