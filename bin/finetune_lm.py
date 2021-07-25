"""


Ver: https://github.com/huggingface/transformers/pull/10255

Obs varias: para la TPU2 no parece haber grandes diferencias de performance entre usar tokenización on the fly

(esto usando una n2-standard-8)
"""

import os
import fire
import random
from glob import glob
from datasets import load_dataset, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from transformers import BertForMaskedLM
from finetune_vs_scratch.preprocessing import special_tokens
from finetune_vs_scratch.model import load_tokenizer
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def tokenize(tokenizer, batch, padding='max_length'):
    return tokenizer(batch['text'], padding=padding, truncation=True, return_special_tokens_mask=True)


def finetune_lm(
    output_dir, num_steps, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    input_dir=None, dataset_path=None, num_files=6, seed=2021,
    batch_size=2048, num_eval_batches=20, limit=None, eval_steps=200, save_steps=1000, padding='max_length',
    per_device_batch_size=32, accumulation_steps=32, warmup_ratio=0.06, weight_decay=0.01, learning_rate=5e-4, on_the_fly=False,
    num_proc=8,
):
    """
    Finetune LM


    """
    print(limit)
    if not input_dir and not dataset_path:
        print("Must provide input_dir or dataset_path")

    random.seed(seed)
    print("Loading model")
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
    tokenizer = load_tokenizer(model_name, 128, model=model)
    print(f"Padding {padding}")

    print("Sanity check")
    print(f"@usuario => {tokenizer.encode('@usuario')}")
    text = "esta es una PRUEBA EN MAYÚSCULAS Y CON TILDES @usuario @usuario"
    print(f"{text} ==> {tokenizer.decode(tokenizer.encode(text))}")

    if input_dir:
        print("Loading datasets")

        tweet_files = random.sample(
            glob(os.path.join(input_dir, "*.txt")),
            num_files
        )

        print(f"Selecting {tweet_files}")

        train_files, test_files = tweet_files[:-1], tweet_files[-1:]

        dataset = load_dataset("text", data_files={"train": train_files, "test": test_files})

    else:
        print(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)

    train_dataset, test_dataset = dataset["train"], dataset["test"]



    if limit:
        print(f"Limiting to {limit}")

        train_dataset = train_dataset.select(list(range(limit)))
        test_dataset = test_dataset.select(list(range(limit)))
    else:
        test_dataset = test_dataset.select(list(range(2048 * num_eval_batches)))

    args = {
        "eval_steps":eval_steps,
        "save_steps":save_steps,
        "logging_steps": 50,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }

    print(args)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )


    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        max_steps=num_steps,

        do_eval= True,
        remove_unused_columns=False,
        logging_dir="./logs",
        logging_strategy="steps",

        **args,
    )



    if on_the_fly:
        with training_args.main_process_first(desc="dataset map tokenization"):
            print("On the fly tokenization")
            train_dataset.set_transform(lambda x: tokenize(tokenizer, x, padding))
            test_dataset.set_transform(lambda x: tokenize(tokenizer, x, padding))
    else:
        print("Tokenization preprocessing")
        print(len(train_dataset))
        print(len(test_dataset))
        batch_size = 2048
        num_proc = 8
        with training_args.main_process_first(desc="dataset map tokenization"):
            print("Tokenizing")
            train_dataset = train_dataset.map(lambda x: tokenize(tokenizer, x, padding), batched=True, batch_size=batch_size, num_proc=num_proc)
            test_dataset = test_dataset.map(lambda x: tokenize(tokenizer, x, padding), batched=True, batch_size=batch_size, num_proc=num_proc)
            train_dataset = train_dataset.remove_columns(["text"])
            test_dataset = test_dataset.remove_columns(["text"])


    # print(train_dataset[0])
    with training_args.main_process_first(desc="Checking lengths"):
        print("Checking lengths")
        print(train_dataset[0])
        train_lengths = {len(ex["input_ids"]) for ex in train_dataset.select(range(20_000))}
        test_lengths = {len(ex["input_ids"]) for ex in test_dataset.select(range(20_000))}
        print(train_lengths)
        print(test_lengths)
        assert len(train_lengths) == 1
        assert len(test_lengths) == 1

    print("Training!")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train(resume_from_checkpoint=None)

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def _mp_fn(*args, **kwargs):
    return fire.Fire(finetune_lm)

if __name__ == '__main__':
    fire.Fire(finetune_lm)