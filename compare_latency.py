import argparse
import os
import time
from typing import Dict, List, Tuple

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import predict


def measure_latency(
    model_func, inputs: List[str], batch_size: int = 32
) -> Tuple[float, float, List[float]]:
    """Measure latency of a model function."""
    latencies = []
    predictions = []

    # Process in batches
    for i in tqdm(range(0, len(inputs), batch_size), desc="Measuring latency"):
        batch = inputs[i : i + batch_size]

        start_time = time.time()
        batch_preds = model_func(batch)
        latency = time.time() - start_time

        # Per-sample latency
        per_sample_latency = latency / len(batch)
        latencies.extend([per_sample_latency] * len(batch))
        predictions.extend(batch_preds)

    mean_latency = np.mean(latencies)
    percentile_95 = np.percentile(latencies, 95)

    return mean_latency, percentile_95, predictions, latencies


def prepare_scenarios(
    dataset_name: str, split: str = "test"
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Prepare datasets with different overlap scenarios."""
    if dataset_name == "paws":
        dataset = load_dataset(dataset_name, "labeled_final", split=split)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        sources = dataset["sentence1"]
        targets = dataset["sentence2"]
    else:
        dataset = load_dataset(dataset_name, split=split)
        sources = dataset["source"]
        targets = dataset["target"]

    # Calculate overlap ratios (simplified heuristic)
    def overlap_ratio(src, tgt):
        src_words = set(word_tokenize(src.lower()))
        tgt_words = set(word_tokenize(tgt.lower()))
        if not src_words or not tgt_words:
            return 0
        return len(src_words & tgt_words) / min(len(src_words), len(tgt_words))

    ratios = [overlap_ratio(s, t) for s, t in zip(sources, targets)]
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0
    std_dev = np.std(ratios) if len(ratios) > 1 else 0.1

    # Set dynamic thresholds based on mean and standard deviation
    high_thresh = min(1.0, mean_ratio + std_dev)
    medium_thresh = mean_ratio
    low_thresh = max(0.0, mean_ratio - std_dev)

    # Ensure we have reasonable separation between categories
    min_separation = 0.05  # Minimum required difference between thresholds
    if high_thresh - medium_thresh < min_separation:
        high_thresh = min(1.0, medium_thresh + min_separation)
    if medium_thresh - low_thresh < min_separation:
        low_thresh = max(0.0, medium_thresh - min_separation)

    # Split into scenarios based on dynamic thresholds
    high_overlap = [
        (s, t) for s, t, r in zip(sources, targets, ratios) if r >= high_thresh
    ]
    medium_overlap = [
        (s, t)
        for s, t, r in zip(sources, targets, ratios)
        if medium_thresh <= r < high_thresh
    ]
    low_overlap = [
        (s, t) for s, t, r in zip(sources, targets, ratios) if r < low_thresh
    ]

    # Take top N samples for each scenario to keep evaluation manageable
    n_samples = min(100, len(high_overlap), len(medium_overlap), len(low_overlap))
    scenarios = {
        "high_overlap": high_overlap[:n_samples],
        "medium_overlap": medium_overlap[:n_samples],
        "low_overlap": low_overlap[:n_samples],
    }

    return scenarios


def evaluate_model(
    model_func, scenarios: Dict[str, Tuple[List[str], List[str]]], model_name: str
) -> pd.DataFrame:
    """Evaluate a model on all scenarios."""
    results = []

    for scenario_name, examples in scenarios.items():
        sources, targets = zip(*examples)
        sources, targets = list(sources), list(targets)

        # Measure latency
        mean_latency, percentile_95, predictions, latencies = measure_latency(
            model_func, sources
        )

        # Calculate iBLEU
        metric = evaluate.load("metrics/my_metric")
        scores = metric.compute(
            sources=sources, predictions=predictions, references=targets
        )
        ibleu = scores["ibleu"]

        results.append(
            {
                "model": model_name,
                "scenario": scenario_name,
                "mean_latency": mean_latency,
                "percentile_95_latency": percentile_95,
                "ibleu": ibleu,
                "all_latencies": latencies,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame):
    """Plot latency vs iBLEU with connected mean and 95th percentile latencies."""
    plt.figure(figsize=(10, 6))

    # Create a color map for models
    models = results_df["model"].unique()
    colors = plt.cm.tab10.colors  # Using matplotlib's default color cycle

    for i, model_name in enumerate(models):
        model_df = results_df[results_df["model"] == model_name]

        # Plot 95th percentile as vertical lines
        for _, row in model_df.iterrows():
            mean_ms = row["mean_latency"] * 1000
            percentile_95_ms = row["percentile_95_latency"] * 1000

            plt.plot(
                [mean_ms, percentile_95_ms],
                [row["ibleu"], row["ibleu"]],
                color=colors[i],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
            )

            # Add vertical line caps for 95th percentile
            plt.scatter(
                percentile_95_ms,
                row["ibleu"],
                color=colors[i],
                marker="|",
                s=200,  # Size of the vertical line
                linewidths=2,
                zorder=2,
            )

            plt.scatter(
                mean_ms,
                row["ibleu"],
                color=colors[i],
                marker="o",
                s=100,  # Size of the point
                zorder=3,
            )

            # Annotate scenario names
            plt.annotate(
                row["scenario"].split("_")[0],
                (mean_ms, row["ibleu"]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
            )

    plt.xlabel("Latency (ms)")
    plt.ylabel("iBLEU")
    # plt.title("Model Performance: Latency vs iBLEU\n(• = mean, | = 95th percentile)")

    # Create custom legend
    legend_elements = []
    for i, model_name in enumerate(models):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=model_name,
                markerfacecolor=colors[i],
                markersize=10,
            )
        )
    plt.legend(handles=legend_elements)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("latency_vs_ibleu.png", dpi=300)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    inputs: List[str],
    max_seq_length: int = 128,
    num_beams: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[str]:
    model = model.to(device)

    # Tokenize inputs
    tokenized = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(device)

    # Generate predictions
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_length=max_seq_length,
            num_beams=num_beams,
        )

    # Decode generated tokens
    predictions = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True,
    )

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--sasar_model_path", type=str, required=True, help="Path to SASAR model"
    )
    parser.add_argument(
        "--sasar_tokenizer", type=str, required=True, help="Tokenizer for SASAR"
    )
    parser.add_argument(
        "--sasar_label_map", type=str, required=True, help="Label map for SASAR"
    )
    parser.add_argument(
        "--t5_model_path", type=str, required=True, help="Path to T5 model"
    )
    parser.add_argument(
        "--t5_tokenizer", type=str, required=True, help="Tokenizer for T5"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun evaluation even if results exist",
    )
    args = parser.parse_args()

    # Check for existing results
    results_file = "latency_ibleu_results.csv"

    if not args.force_rerun and os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        all_results = pd.read_csv(results_file)

        # Convert string representation of list to actual list for all_latencies
        if "all_latencies" in all_results.columns:
            all_results["all_latencies"] = all_results["all_latencies"].apply(
                lambda x: [float(i) for i in x.strip("[]").split(",") if i.strip()]
            )
    else:
        # Prepare scenarios
        scenarios = prepare_scenarios(args.dataset_name)

        # Initialize models

        # SASAR predictor
        sasar_predictor = predict.JointPredictor(
            model_filepath=args.sasar_model_path,
            tokenizer_name=args.sasar_tokenizer,
            label_map_file=args.sasar_label_map,
            sequence_length=args.max_seq_length,
            use_open_vocab=True,
            is_pointing=True,
            special_glue_string_for_joining_sources="[SEP]",
            use_token_type_ids=True,
            no_deleted_spans=True,
        )

        # Calculate SASAR model size
        sasar_model_size = sasar_predictor.get_number_of_parameters() / 1e6
        print(f"\nSASAR Model Size: {sasar_model_size:.2f}M parameters")

        def sasar_predict(sources: List[str]) -> List[str]:
            """Wrapper for SASAR prediction."""
            _, predicted_inserts = sasar_predictor.predict_end_to_end_batch(sources)
            return predicted_inserts

        # T5 predictor
        t5_model, t5_tokenizer = load_model_and_tokenizer(
            args.t5_model_path, args.t5_tokenizer
        )

        # Calculate T5 model size
        t5_total_params = sum(p.numel() for p in t5_model.parameters())
        t5_model_size = t5_total_params / 1e6
        print(f"T5 Model Size: {t5_model_size:.2f}M parameters\n")

        def t5_predict(sources: List[str]) -> List[str]:
            """Wrapper for T5 prediction."""
            return generate_predictions(
                t5_model, t5_tokenizer, sources, args.max_seq_length
            )

        # Evaluate both models
        print("Running evaluations...")
        sasar_results = evaluate_model(sasar_predict, scenarios, "SASAR")
        t5_results = evaluate_model(t5_predict, scenarios, "T5")

        # Combine results
        all_results = pd.concat([sasar_results, t5_results], ignore_index=True)

        # Save detailed results
        all_results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

    # Print results
    print("\nEvaluation Results:")
    print(
        all_results[
            ["model", "scenario", "mean_latency", "percentile_95_latency", "ibleu"]
        ]
    )

    # Plot results
    plot_results(all_results)


if __name__ == "__main__":
    main()
