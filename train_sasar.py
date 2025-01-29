import argparse
import json
import logging
import math
import os

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

import my_models
import utils
from data_collator import DataCollatorForTagging

logger = get_logger(__name__)


class TextEditingDataset(Dataset):
    """Text Editing dataset."""

    def __init__(self, data_file=None, is_insertion=False):
        if data_file is None:
            raise ValueError("Please provide a data file.")

        data_list = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                example_dict = json.loads(line)
                data_list.append(example_dict)

        self.data = data_list
        self.is_insertion = is_insertion

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.is_insertion:
            return {
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(data["input_mask"], dtype=torch.long),
                "labels": torch.tensor(data["masked_lm_ids"], dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(data["input_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(
                    data["token_type_ids"], dtype=torch.long
                ),
                "edit_tags": torch.tensor(data["labels"], dtype=torch.long),
                "pointers": torch.tensor(data["point_indexes"], dtype=torch.long),
                "labels_mask": torch.tensor(data["labels_mask"], dtype=torch.float32),
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Train text editing model")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A file containing the validation data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--use_open_vocab",
        action="store_true",
        help="Currently only use_open_vocab=True is supported",
    )
    parser.add_argument(
        "--train_insertion",
        action="store_true",
        help="Whether to train an inserter (True) or a tagger (False)",
    )
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, help="Number of training epochs"
    )
    parser.add_argument("--learning_rate", type=float, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="The number of epochs to wait for the validation loss to improve before stopping training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Whether to use pseudo semantic graphs for the tagger model",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--label_map_file", type=str, required=True)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--use_pointing", action="store_true")
    parser.add_argument("--pointing_weight", type=float, default=1.0)
    parser.add_argument("--mini_epochs_per_epoch", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--use_weighted_labels",
        action="store_true",
        help="Whether different labels were given different weights. Primarly used to increase the importance of rare tags. Only True is currently supported.",
    )

    args = parser.parse_args()

    if not args.use_open_vocab:
        raise ValueError("Currently only `use_open_vocab=True` is supported")

    if args.with_graph and args.train_insertion:
        raise ValueError("Only tagging models are supported with graphs")

    return args


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    label_list = utils.read_label_map(args.label_map_file, use_str_keys=True)
    label_list = {v: k for k, v in label_list.items()}

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.pointing = args.use_pointing
    config.num_classes = len(label_list)
    config.query_size = 64
    config.query_transformer = True
    if args.train_insertion:
        model = my_models.get_insertion_model(config)
    else:
        model = my_models.get_tagging_model(
            config,
            args.max_seq_length,
            pointing_weight=args.pointing_weight,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if accelerator.is_main_process:
        # Handle the repository creation
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Load the data from the file
        if args.train_file is not None:
            train_dataset = TextEditingDataset(
                args.train_file, is_insertion=args.train_insertion
            )
        if args.validation_file is not None:
            validation_dataset = TextEditingDataset(
                args.validation_file, is_insertion=args.train_insertion
            )

        label_pad_token_id = -100

        if args.train_insertion:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
            )
        else:
            data_collator = DataCollatorForTagging(
                tokenizer,
                pad_to_multiple_of=(
                    8 if accelerator.mixed_precision == "fp16" else None
                ),
            )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = DataLoader(
            validation_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
    accelerator.wait_for_everyone()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    metric = evaluate.load("seqeval")

    def get_labels(predictions, references):
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

        # # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != 0]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != 0]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_stepp

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    early_stopping_patience = args.patience
    best_metric = None
    patience_counter = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs[0]
            # We keep track of the loss at each epoch
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        current_metric = None
        if args.train_insertion:
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(
                        accelerator.gather_for_metrics(
                            loss.repeat(args.per_device_eval_batch_size)
                        )
                    )

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            accelerator.print(f"epoch {epoch}:", {"perplexity": perplexity})
            current_metric = eval_loss
        else:
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                tag_logits, pointing_logits = outputs
                predicted_tags = torch.argmax(tag_logits, dim=-1)
                true_tags = batch["edit_tags"]
                predicted_pointers = torch.argmax(pointing_logits, dim=-1)
                true_pointers = batch["pointers"]

                preds, refs = get_labels(predicted_tags, true_tags)

                metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids

            eval_metric = compute_metrics()
            accelerator.print(f"epoch {epoch}:", eval_metric)
            current_metric = eval_metric["f1"]

        if early_stopping_patience is not None:
            if (
                best_metric is None
                or (args.train_insertion and current_metric < best_metric)
                or (not args.train_insertion and current_metric > best_metric)
            ):
                best_metric = current_metric
                patience_counter = 0
                # Save the best model
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        os.path.join(args.output_dir, "best_model"),
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
            else:
                patience_counter += 1
                accelerator.print(
                    f"No improvement in metric. Patience counter: {patience_counter}/{early_stopping_patience}"
                )

            if patience_counter >= early_stopping_patience:
                accelerator.print("Early stopping triggered!")
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
            if args.train_insertion:
                json.dump({"perplexity": perplexity}, f)
            else:
                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                # Convert all float64 & int64 type numbers to float & int for json serialization
                for key, value in all_results.items():
                    if isinstance(value, np.float64):
                        all_results[key] = float(value)
                    elif isinstance(value, np.int64):
                        all_results[key] = int(value)
                json.dump(all_results, f)


if __name__ == "__main__":
    main()
