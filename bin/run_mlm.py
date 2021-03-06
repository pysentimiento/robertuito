#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
from multiprocessing.sharedctypes import Value
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from glob import glob
import datasets
import tempfile
from datasets import load_dataset
import random
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from finetune_vs_scratch.dataset import DistributedBatchProcessedDataset, DummyDataset, BatchProcessedDataset, JSONDistributedBatchProcessedDataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )



    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )



    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train files."}
    )
    eval_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test files."}
    )


    use_datasets: bool = field(
        default=False,
        metadata={
            "help": "Use `datasets` instead of our custom datasets",
        },
    )


    use_cached_dataset: bool = field(
        default=False,
        metadata={
            "help": "Use cached dataset on multiprocess setting",
        },
    )


    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    tokenization_batch_size: Optional[int] = field(
        default=4096, metadata={
            "help": "Tokenization batch size (for on-the-fly tokenization)"
        }
    )

    tokenize_on_the_fly: bool = field(
        default=False, metadata={
            "help": "Tokenize on-the-fly"
        }
    )

    dummy_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Load dummy dataset for testing purposes"
        }
    )

    def __post_init__(self):
        if not self.dummy_dataset and (self.train_dir is None or self.eval_dir is None):
            raise ValueError("Must provide train_dir & eval_dir or set dummy_dataset")


def main(seed):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Set seed before initializing model.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if (2 <= len(sys.argv) <= 4):
        if sys.argv[-1].endswith(".json"):
            json_file = sys.argv[-1]
        else:
            json_file = sys.argv[1]
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(json_file))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    """
    This seed is only to shuffle lists
    """
    log_level = training_args.get_process_log_level()
    print(f"Logger level: {log_level}")
    logger.setLevel(log_level)

    if data_args.train_dir:
        logger.info(f"Seed : {seed}")
        random.seed(seed)
        train_files = sorted(glob(os.path.join(data_args.train_dir, "*.txt*")))
        eval_files = sorted(glob(os.path.join(data_args.eval_dir, "*.txt*")))

        random.shuffle(train_files)

        logger.info(f"First files: {train_files[:5]}")
        logger.info(f"Eval files: {eval_files[:5]}")





    set_seed(training_args.seed)
    print("Model args: ",model_args)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    print("World size: ", training_args.world_size)
    print("Local Process index: ", training_args.local_process_index)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.model_max_length =128
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    if data_args.dummy_dataset:
        logger.info("Dummy dataset")
        train_dataset = DummyDataset(
            tokenizer("Esto es una prueba @usuario", padding=padding, truncation=True, return_special_tokens_mask=True),
            50_000_000
        )

        eval_dataset = train_dataset
        training_args.remove_unused_columns = False
    else:
        """
        Load train and test
        """
        if data_args.use_datasets:

            """
            Load using datasets
            """

            raw_datasets = load_dataset("text",
                data_files = {"train": train_files, "test": eval_files}
            )
        else:
            """
            Use our custom class
            """
            training_args.remove_unused_columns = False

            if data_args.max_eval_samples is None or not type(data_args.max_eval_samples) is int:
                raise ValueError("Must provide max_eval_samples")
            logger.info(f"Tokenization batch size {data_args.tokenization_batch_size}")

            if training_args.world_size > 1 and data_args.use_cached_dataset:
                """
                Use distributed batch processed to make this faster
                """

                cache_file = f"mlm-cache.json"
                test_cache_file = f"/home/jmperez/mlm-cache-test.json"
                is_master = training_args.local_process_index == 0

                logger.info(f"Using distributed dataset")
                logger.info(f"Cache file name: {cache_file}")
                logger.info(f"Test Cache file name: {test_cache_file}")
                logger.info(f"Is master?     : {is_master}")

                train_dataset = JSONDistributedBatchProcessedDataset(
                    train_files, tokenizer, is_master=is_master, cache_file=cache_file,
                    batch_size=data_args.tokenization_batch_size,
                    padding=padding,
                )
                # I use the normal option because
                # for some reason it won't work because of Trainer's dataloaders

                # TODO: I limit this to 4; move
                eval_dataset = load_dataset("text", data_files={"test": eval_files[:5]})["test"]

                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
                # eval_dataset = BatchProcessedDataset(
                #     eval_files, tokenizer,
                #     batch_size=data_args.tokenization_batch_size,
                #     padding=padding, limit=data_args.max_eval_samples
                # )

            else:
                logger.info("Using BatchProcessedDataset")
                train_dataset = BatchProcessedDataset(
                    train_files, tokenizer, batch_size=data_args.tokenization_batch_size,
                    padding=padding,
                )
                eval_dataset = BatchProcessedDataset(
                    eval_files, tokenizer, batch_size=data_args.tokenization_batch_size,
                    padding=padding, limit=data_args.max_eval_samples
                )


    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    if data_args.use_datasets:
        logger.info("Using datasets library")

        if data_args.tokenize_on_the_fly:
            raw_datasets.set_transform(tokenize_function)
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["test"]

        else:
            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=["text"],
                    #load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )

            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets["test"]
    elif data_args.use_cached_dataset:
        logger.info("Tokenizing test dataset")

        with training_args.main_process_first(desc="dataset map tokenization"):
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["text"],
                #load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset line_by_line",
            )


    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics


        try:
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        except TypeError:
            # TPU =>
            metrics["train_samples"] = None

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        try:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        except:
            metrics["train_samples"] = None

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index, seed):
    # For xla_spawn (TPUs)
    main(seed)


if __name__ == "__main__":
    main(time.time())