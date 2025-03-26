import argparse
import re

import evaluate
import nltk
import pandas as pd

# Download NLTK package if needed
nltk.download("punkt")


def normalize_text(text):
    """Removes special tokens, lowercases, and trims text."""
    if pd.isna(text):
        return ""
    text = re.sub(r"\[CLS\]|\[SEP\]", "", text)  # Remove special tokens
    text = text.lower().strip()  # Lowercase and trim
    return text


def determine_columns(df):
    """Determine which column combination exists in the dataframe."""
    if "prediction" in df.columns and "reference" in df.columns:
        return ("prediction", "reference")
    elif (
        "predicted_tags" in df.columns
        and "predicted_insertions" in df.columns
        and "target" in df.columns
    ):
        return ("predicted_insertions", "target")
    else:
        raise ValueError(
            "TSV file must contain either:\n"
            "1. 'source', 'prediction', 'reference' columns\n"
            "2. 'source', 'predicted_tags', 'predicted_insertions', 'target' columns"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU or SBERT similarity for predictions."
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the .tsv file"
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to compute",
    )
    args = parser.parse_args()

    # Read TSV file
    df = pd.read_csv(args.file_path, sep="\t")

    pred_col, ref_col = determine_columns(df)

    # Normalize text
    df[pred_col] = df[pred_col].apply(normalize_text)
    df[ref_col] = df[ref_col].apply(normalize_text)
    df["source"] = df["source"].apply(normalize_text)

    metric = evaluate.load(args.metric)

    results = metric.compute(
        sources=df["source"], predictions=df[pred_col], references=df[ref_col]
    )
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
