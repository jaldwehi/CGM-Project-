from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def parse_glucose_events(xml_path: Path) -> pd.DataFrame:
    """
    Parse OhioT1DM-style XML:
      <patient ...>
        <glucose_level>
          <event ts="30-11-2021 17:06:00" value="160"/>
          ...

    Returns DataFrame with columns: timestamp, glucose
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    glucose_node = root.find("glucose_level")
    if glucose_node is None:
        raise ValueError(f"Could not find <glucose_level> in {xml_path.name}")

    rows = []
    for ev in glucose_node.findall("event"):
        ts = ev.get("ts")
        val = ev.get("value")
        if ts is None or val is None:
            continue
        rows.append((ts, val))

    df = pd.DataFrame(rows, columns=["timestamp_raw", "glucose_raw"])

    # OhioT1DM timestamps often look like: "30-11-2021 17:06:00"  (day-first)
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], dayfirst=True, errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose_raw"], errors="coerce")

    df = df.dropna(subset=["timestamp", "glucose"]).copy()
    df = df[["timestamp", "glucose"]].sort_values("timestamp").reset_index(drop=True)
    return df


def build_merged_cgm_csv(training_xml: Path, testing_xml: Path, out_csv: Path) -> pd.DataFrame:
    train_df = parse_glucose_events(training_xml)
    test_df = parse_glucose_events(testing_xml)

    # Merge + sort + drop duplicates on timestamp (there may be overlap)
    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Nightscout-friendly columns
    full["sgv"] = full["glucose"].round(0).astype(int)
    full["date"] = (full["timestamp"].astype("int64") // 10**6).astype("int64")  # epoch ms
    full["dateString"] = full["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False)
    return full


if __name__ == "__main__":
    training = Path("data/591-ws-training.xml")
    testing  = Path("data/591-ws-testing.xml")
    out_csv  = Path("data/cgm_591.csv")

    df = build_merged_cgm_csv(training, testing, out_csv)
    print("Saved:", out_csv)
    print("Rows:", len(df))
    if len(df):
        print("From:", df["timestamp"].min(), "To:", df["timestamp"].max())
