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
from torch import nn
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
from data_collator import DataCollatorForJointModel, DataCollatorForTagging

logger = get_logger(__name__)


class TextEditingDataset(Dataset):
    """Text Editing dataset."""

    def __init__(self, data_file=None, model_type="tagger", use_token_type_ids=True):
        if data_file is None:
            raise ValueError("Please provide a data file.")

        data_list = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                example_dict = json.loads(line)
                data_list.append(example_dict)

        self.data = data_list
        self.model_type = model_type
        self.use_token_type_ids = use_token_type_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.model_type == "inserter":
            item = {
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(data["input_mask"], dtype=torch.long),
            }
            item["labels"] = torch.tensor(data["masked_lm_ids"], dtype=torch.long)
        elif self.model_type == "tagger":
            item = {
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(data["input_mask"], dtype=torch.long),
            }
            if self.use_token_type_ids:
                item["token_type_ids"] = torch.tensor(
                    data["token_type_ids"], dtype=torch.long
                )
            item["edit_tags"] = torch.tensor(data["labels"], dtype=torch.long)
            item["pointers"] = torch.tensor(data["point_indexes"], dtype=torch.long)
            item["labels_mask"] = torch.tensor(data["labels_mask"], dtype=torch.float32)
        elif self.model_type == "joint":
            item = {
                # Tagging task inputs
                "tagging_input_ids": torch.tensor(
                    data["tagging_input_ids"], dtype=torch.long
                ),
                "tagging_input_mask": torch.tensor(
                    data["tagging_input_mask"], dtype=torch.long
                ),
                "tagging_token_type_ids": torch.tensor(
                    data["tagging_token_type_ids"], dtype=torch.long
                ),
                "pointers": torch.tensor(data["point_indexes"], dtype=torch.long),
                "edit_tags": torch.tensor(data["labels"], dtype=torch.long),
                "labels_mask": torch.tensor(data["labels_mask"], dtype=torch.float32),
                # Insertion task inputs
                "insertion_input_ids": torch.tensor(
                    data["insertion_input_ids"], dtype=torch.long
                ),
                "insertion_input_mask": torch.tensor(
                    data["insertion_input_mask"], dtype=torch.long
                ),
                "insertion_token_type_ids": torch.tensor(
                    data["insertion_token_type_ids"], dtype=torch.long
                ),
                "masked_lm_ids": torch.tensor(data["masked_lm_ids"], dtype=torch.long),
            }

        return item


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
        "--model_type",
        type=str,
        default="tagger",
        choices=["tagger", "inserter", "joint"],
        help="Type of model to train: tagger, inserter, or joint.",
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
    parser.add_argument(
        "--use_token_type_ids",
        action="store_true",
        help="Whether to use token_type_ids in the dataset",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

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

    if "with_graph" in args.train_file:

        def extract_relations_from_amr(file_path):
            relations = set()
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example_dict = json.loads(line)
                    amr_source = example_dict.get("amr_source", "")
                    # Extract words starting with :
                    relations.update(
                        word for word in amr_source.split() if word.startswith(":")
                    )
            return list(relations)

        def load_or_extract_relations():
            relations_file = os.path.join("cache_files", "amr_relations.json")
            if os.path.exists(relations_file):
                with open(relations_file, "r", encoding="utf-8") as f:
                    relations = json.load(f)
            else:
                train_relations = extract_relations_from_amr(
                    "cache_files/paws_AMR_train.jsonl"
                )
                relations = list(set(train_relations))
                with open(relations_file, "w", encoding="utf-8") as f:
                    json.dump(relations, f)
            return relations

        amr_relations = load_or_extract_relations()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, additional_special_tokens=amr_relations
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.model_type == "inserter":
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.vocab_size = len(tokenizer)
        model = my_models.get_insertion_model(config, args.model_name_or_path)
    elif args.model_type == "tagger":
        model = my_models.get_tagging_model(
            args.model_name_or_path,
            args.max_seq_length,
            pointing_weight=args.pointing_weight,
            use_pointing=args.use_pointing,
            num_classes=len(label_list),
            vocab_size=len(tokenizer),
        )
    elif args.model_type == "joint":
        model = my_models.get_joint_model(
            args.model_name_or_path,
            args.max_seq_length,
            pointing_weight=args.pointing_weight,
            use_pointing=args.use_pointing,
            num_classes=len(label_list),
            vocab_size=len(tokenizer),
        )
    else:
        raise ValueError(f"Model type {args.model_type} not recognized.")

    if "with_graph" in args.train_file:
        model.resize_token_embeddings(len(tokenizer))

    if accelerator.is_main_process:
        # Handle the repository creation
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Load the data from the file
        if args.train_file is not None:
            train_dataset = TextEditingDataset(
                args.train_file,
                model_type=args.model_type,
                use_token_type_ids=args.use_token_type_ids,
            )
        if args.validation_file is not None:
            validation_dataset = TextEditingDataset(
                args.validation_file,
                model_type=args.model_type,
                use_token_type_ids=args.use_token_type_ids,
            )

        label_pad_token_id = -100

        if args.model_type == "inserter":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
            )
        elif args.model_type == "tagger":
            data_collator = DataCollatorForTagging(
                tokenizer,
                pad_to_multiple_of=(
                    8 if accelerator.mixed_precision == "fp16" else None
                ),
            )
        elif args.model_type == "joint":
            data_collator = DataCollatorForJointModel(
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
    best_metrics = None
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
        if args.model_type == "inserter":
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
        elif args.model_type == "tagger":
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
        elif args.model_type == "joint":
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    tag_logits, pointing_logits, mlm_logits = model(**batch)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        mlm_logits.view(-1, mlm_logits.size(-1)),
                        batch["masked_lm_ids"].view(-1),
                    )
                    losses.append(
                        accelerator.gather_for_metrics(
                            loss.repeat(args.per_device_eval_batch_size)
                        )
                    )

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
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            accelerator.print(
                f"epoch {epoch}:", eval_metric, {"perplexity": perplexity}
            )
            current_metric = (eval_metric["f1"] + (1 / perplexity)) / 2

        if args.model_type == "inserter":
            current_metrics = {"perplexity": perplexity}
        elif args.model_type == "tagger":
            current_metrics = {
                f"eval_{k}": (
                    float(v)
                    if isinstance(v, (np.float64, np.float32))
                    else int(v) if isinstance(v, (np.int64, np.int32)) else v
                )
                for k, v in eval_metric.items()
            }
        elif args.model_type == "joint":
            current_metrics = {
                f"eval_{k}": (
                    float(v)
                    if isinstance(v, (np.float64, np.float32))
                    else int(v) if isinstance(v, (np.int64, np.int32)) else v
                )
                for k, v in eval_metric.items()
            }
            current_metrics["perplexity"] = perplexity

        if early_stopping_patience is not None:
            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_metrics = current_metrics
                patience_counter = 0
                # Save the best model
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if "with_graph" in args.train_file:
                        tokenizer.save_pretrained(args.output_dir)
            else:
                patience_counter += 1
                accelerator.print(
                    f"No improvement in metric. Patience counter: {patience_counter}/{early_stopping_patience}"
                )

            if patience_counter >= early_stopping_patience:
                accelerator.print("Early stopping triggered!")
                if args.output_dir is not None and best_metrics is not None:
                    with open(
                        os.path.join(args.output_dir, "eval_scores.json"), "w"
                    ) as f:
                        json.dump(best_metrics, f)
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # If we finished all epochs without early stopping, save metrics now
    if (
        args.output_dir is not None
        and best_metrics is not None
        and (
            early_stopping_patience is None
            or patience_counter < early_stopping_patience
        )
    ):
        with open(os.path.join(args.output_dir, "eval_scores.json"), "w") as f:
            json.dump(best_metrics, f)


if __name__ == "__main__":
    main()
