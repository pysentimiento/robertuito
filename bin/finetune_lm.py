import os
import torch
import fire
import datasets
from datasets import load_from_disk
import torch
from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from torch.utils.data.dataloader import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer
from finetune_vs_scratch.preprocessing import special_tokens
from finetune_vs_scratch.model import load_model_and_tokenizer, load_tokenizer

def finetune_lm(
    output_dir, train_path, test_path, num_steps, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    batch_size=2048, num_eval_batches=50, deepspeed=None, limit=None, eval_and_save_steps=100, per_device_batch_size=32,
    accumulation_steps=32, warmup_ratio=0.06, weight_decay=0.01, learning_rate=5e-4, on_the_fly=False,
):
    """
    Finetune LM
    """

    print("Loading datasets")

    train_dataset = load_from_disk(train_path)
    test_dataset = load_from_disk(test_path)

    if limit:
        print(f"Limiting to {limit}")

        train_dataset = train_dataset.select(list(range(limit)))
        test_dataset = test_dataset.select(list(range(limit)))
    else:
        tweets_needed = num_steps * batch_size

        print(f"Subselecting {tweets_needed}")
        train_dataset = train_dataset.select(list(range(tweets_needed)))
        test_dataset = test_dataset.select(list(range(batch_size * num_eval_batches)))


    print("Loading model")
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
    tokenizer = load_tokenizer(model_name, 128, model=model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    print("Tokenizing")

    def tokenize(batch):
        return tokenizer(batch['text'], padding=False, truncation=True, return_special_tokens_mask=True)

    if on_the_fly:
        train_dataset.set_transform(tokenize)
        test_dataset.set_transform(tokenize)
    else:
        train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=24)
        test_dataset = test_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=24)
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

    args = {
        "eval_steps":eval_and_save_steps,
        "save_steps":eval_and_save_steps,
        "logging_steps": 50,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "deepspeed": deepspeed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }

    print(args)

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