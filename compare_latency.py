import argparse
import time
from typing import Dict, List, Tuple

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load

# Import T5 prediction code (assuming it's in t5_predict.py)
from t5_predict import generate_predictions, load_model_and_tokenizer
from tqdm import tqdm

from predict import batch_generator, get_predictor


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
        src_words = set(src.split())
        tgt_words = set(tgt.split())
        if not src_words or not tgt_words:
            return 0
        return len(src_words & tgt_words) / min(len(src_words), len(tgt_words))

    ratios = [overlap_ratio(s, t) for s, t in zip(sources, targets)]

    # Split into scenarios based on overlap ratio
    high_overlap = [(s, t) for s, t, r in zip(sources, targets, ratios) if r > 0.7]
    medium_overlap = [
        (s, t) for s, t, r in zip(sources, targets, ratios) if 0.3 <= r <= 0.7
    ]
    low_overlap = [(s, t) for s, t, r in zip(sources, targets, ratios) if r < 0.3]

    # Take top N samples for each scenario to keep evaluation manageable
    n_samples = min(200, len(high_overlap), len(medium_overlap), len(low_overlap))
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
        results = metric.compute(
            sources=sources, predictions=predictions, references=targets
        )
        ibleu = results["ibleu"]

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
    """Plot latency vs iBLEU for all models and scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mean latency
    for model_name in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model_name]
        ax1.plot(model_df["mean_latency"], model_df["ibleu"], "o-", label=model_name)
        for i, row in model_df.iterrows():
            ax1.annotate(row["scenario"], (row["mean_latency"], row["ibleu"]))

    ax1.set_xlabel("Mean Latency (seconds)")
    ax1.set_ylabel("iBLEU Score")
    ax1.set_title("Mean Latency vs iBLEU")
    ax1.legend()
    ax1.grid(True)

    # Plot 95th percentile latency
    for model_name in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model_name]
        ax2.plot(
            model_df["percentile_95_latency"], model_df["ibleu"], "o-", label=model_name
        )
        for i, row in model_df.iterrows():
            ax2.annotate(row["scenario"], (row["percentile_95_latency"], row["ibleu"]))

    ax2.set_xlabel("95th Percentile Latency (seconds)")
    ax2.set_ylabel("iBLEU Score")
    ax2.set_title("95th Percentile Latency vs iBLEU")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("latency_vs_ibleu.png")
    plt.show()


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
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=128, help="Max sequence length"
    )
    args = parser.parse_args()

    # Prepare scenarios
    scenarios = prepare_scenarios(args.dataset_name)

    # Initialize models

    # SASAR predictor
    sasar_predictor = get_predictor(
        args.sasar_model_path, args.sasar_tokenizer, args.sasar_label_map, args
    )

    def sasar_predict(sources: List[str]) -> List[str]:
        """Wrapper for SASAR prediction."""
        _, predicted_inserts = sasar_predictor.predict_end_to_end_batch(sources)
        return predicted_inserts

    # T5 predictor
    t5_model, t5_tokenizer = load_model_and_tokenizer(args.t5_model_path)

    def t5_predict(sources: List[str]) -> List[str]:
        """Wrapper for T5 prediction."""
        return generate_predictions(
            t5_model, t5_tokenizer, sources, args.max_seq_length
        )

    # Evaluate both models
    sasar_results = evaluate_model(sasar_predict, scenarios, "SASAR")
    t5_results = evaluate_model(t5_predict, scenarios, "T5")

    # Combine results
    all_results = pd.concat([sasar_results, t5_results], ignore_index=True)

    # Print results
    print("\nEvaluation Results:")
    print(
        all_results[
            ["model", "scenario", "mean_latency", "percentile_95_latency", "ibleu"]
        ]
    )

    # Plot results
    plot_results(all_results)

    # Save detailed results
    all_results.to_csv("latency_ibleu_results.csv", index=False)


if __name__ == "__main__":
    main()
