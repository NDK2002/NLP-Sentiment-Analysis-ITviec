from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotation"
SAMPLE_PER_CLASS = 100
RANDOM_STATE = 2026


def main() -> None:
    data = pd.read_excel(SOURCE)
    sampled = (
        data.groupby("sentiment", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index()
        .rename(columns={"index": "source_index"})
    )
    sampled.insert(0, "sample_id", [f"SA-{index:03d}" for index in range(1, len(sampled) + 1)])

    blind = sampled[["sample_id", "raw_review_text"]].copy()
    blind.columns = ["sample_id", "review_text"]
    blind["annotator_1_label"] = ""
    blind["annotator_2_label"] = ""
    blind["adjudicated_label"] = ""
    blind["notes"] = ""

    key = sampled[
        [
            "sample_id",
            "source_index",
            "id",
            "Company Name",
            "Rating",
            "sentiment",
            "Recommend?",
        ]
    ].copy()
    key.columns = [
        "sample_id",
        "source_index",
        "id",
        "Company Name",
        "Rating",
        "rating_derived_weak_label",
        "Recommend?",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blind.to_csv(OUTPUT_DIR / "sentiment_audit_blind.csv", index=False)
    key.to_csv(OUTPUT_DIR / "sentiment_audit_key.csv", index=False)
    print(f"Created {len(blind)} blind annotation rows and a separate rating key.")


if __name__ == "__main__":
    main()
