import argparse
import re

import nltk
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sentence_transformers import SentenceTransformer, util

# Download NLTK package if needed
nltk.download("punkt")


def normalize_text(text):
    """Removes special tokens, lowercases, and trims text."""
    if pd.isna(text):
        return ""
    text = re.sub(r"\[CLS\]|\[SEP\]", "", text)  # Remove special tokens
    text = text.lower().strip()  # Lowercase and trim
    return text


# def compute_bleu(predictions, references):
#     """Computes BLEU score using NLTK with smoothing."""
#     smoothie = SmoothingFunction().method4
#     bleu_scores = [
#         sentence_bleu(
#             [ref.split()], pred.split(), smoothing_function=smoothie  # Tokenized input
#         )
#         for pred, ref in zip(predictions, references)
#     ]
#     return sum(bleu_scores) / len(bleu_scores)


def compute_bleu(predictions, references):
    """Computes BLEU score using corpus-level BLEU."""
    smoothie = SmoothingFunction().method4
    references = [[ref.split()] for ref in references]  # List of lists for corpus_bleu
    predictions = [pred.split() for pred in predictions]  # Tokenized predictions
    return corpus_bleu(references, predictions, smoothing_function=smoothie)


def compute_sbert_similarity(predictions, references, model):
    """Computes average cosine similarity using SBERT embeddings."""
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    similarities = util.cos_sim(pred_embeddings, ref_embeddings).diagonal()
    return similarities.mean().item()


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
        choices=["bleu", "sbert"],
        required=True,
        help="Metric to compute",
    )
    args = parser.parse_args()

    # Read TSV file
    df = pd.read_csv(args.file_path, sep="\t")

    if "predicted_insertions" not in df.columns or "target" not in df.columns:
        raise ValueError(
            "TSV file must contain 'predicted_insertions' and 'target' columns."
        )

    # Normalize text
    df["predicted_insertions"] = df["predicted_insertions"].apply(normalize_text)
    df["target"] = df["target"].apply(normalize_text)
    df["source"] = df["source"].apply(normalize_text)

    # Compute the requested metric
    if args.metric == "bleu":
        bleu_pred_target = compute_bleu(
            df["predicted_insertions"].tolist(), df["target"].tolist()
        )
        bleu_source_target = compute_bleu(df["source"].tolist(), df["target"].tolist())

        print(f"BLEU (Predicted vs. Target): {bleu_pred_target:.4f}")
        print(f"BLEU (Source vs. Target): {bleu_source_target:.4f}")
    elif args.metric == "sbert":
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Load SBERT model

        sbert_pred_target = compute_sbert_similarity(
            df["predicted_insertions"].tolist(), df["target"].tolist(), model
        )
        sbert_source_target = compute_sbert_similarity(
            df["source"].tolist(), df["target"].tolist(), model
        )

        print(f"SBERT Similarity (Predicted vs. Target): {sbert_pred_target:.4f}")
        print(f"SBERT Similarity (Source vs. Target): {sbert_source_target:.4f}")
    else:
        raise ValueError("Invalid metric")


if __name__ == "__main__":
    main()
