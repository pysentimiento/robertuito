import os
import fire
from glob import glob
from datasets import load_dataset, Features, Value, load_from_disk, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from transformers import BertForMaskedLM
from finetune_vs_scratch.preprocessing import special_tokens
from finetune_vs_scratch.model import load_tokenizer
import torch_xla.core.xla_model as xm

def tokenize(tokenizer, batch, padding='max_length'):
    return tokenizer(batch['text'], padding=padding, truncation=True, return_special_tokens_mask=True)


def train(model, tokenizer, train_dataset, test_dataset, output_dir, num_steps, on_the_fly=False, **kwargs):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    if on_the_fly:
        print("On the fly tokenization")
        train_dataset.set_transform(lambda x: tokenize(tokenizer, x))
        test_dataset.set_transform(lambda x: tokenize(tokenizer, x))
        print(train_dataset[0])
        print(len(train_dataset[1000]["input_ids"]))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        max_steps=num_steps,


        do_eval= True,
        remove_unused_columns=False,
        logging_dir="./logs",
        logging_strategy="steps",

        **kwargs,
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

def _mp_fn(index, model_name, dataset_path, output_dir, num_steps, num_eval_batches, training_args):
    print("Loading model...")
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
    tokenizer = load_tokenizer(model_name, 128, model=model)
    print(f"Loading from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print("Done")
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    test_dataset = test_dataset.select(list(range(2048 * num_eval_batches)))

    return train(model, tokenizer, train_dataset, test_dataset, output_dir, num_steps, **training_args)

def finetune_lm(
    output_dir, num_steps, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    input_dir=None, dataset_path=None,
    batch_size=2048, num_eval_batches=20, deepspeed=None, limit=None, eval_steps=200, save_steps=1000,
    per_device_batch_size=32, accumulation_steps=32, warmup_ratio=0.06, weight_decay=0.01, learning_rate=5e-4, on_the_fly=False,
    num_tpu_cores=None,
):
    """
    Finetune LM


    """
    if not input_dir and not dataset_path:
        print("Must provide input_dir or dataset_path")


    print("Loading model")
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
    tokenizer = load_tokenizer(model_name, 128, model=model)
    padding = 'max_length' if num_tpu_cores else False
    print(f"Padding {padding}")

    print("Sanity check")
    print(f"@usuario => {tokenizer.encode('@usuario')}")
    text = "esta es una PRUEBA EN MAYÚSCULAS Y CON TILDES @usuario @usuario"
    print(f"{text} ==> {tokenizer.decode(tokenizer.encode(text))}")

    if input_dir:
        print("Loading datasets")

        tweet_files = sorted(
            glob(os.path.join(input_dir, "*.txt"))
        )

        train_files = tweet_files[:10]
        test_files = tweet_files[-1:]

        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")
        features = Features({
            'text': Value('string'),
        })

        train_dataset, test_dataset = load_dataset(
            "text", data_files={"train": train_files, "test": test_files}, split=["train", "test"], features=features
        )
    else:
        dataset = load_from_disk(dataset_path)
        train_dataset, test_dataset = dataset["train"], dataset["test"]


        if limit:
            print(f"Limiting to {limit}")

            train_dataset = train_dataset.select(list(range(limit)))
            test_dataset = test_dataset.select(list(range(limit)))


        """
        If TPU => pad to max length

        See https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#known-performance-caveats
        """

    if not on_the_fly:
        print("Tokenizing")
        train_dataset = train_dataset.map(lambda x: tokenize(tokenizer, x), batched=True, batch_size=batch_size)
        test_dataset = test_dataset.map(lambda x: tokenize(tokenizer, x), batched=True, batch_size=batch_size)
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    args = {
        "eval_steps":eval_steps,
        "save_steps":save_steps,
        "logging_steps": 50,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "deepspeed": deepspeed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "on_the_fly": on_the_fly,
    }

    print(args)

    if num_tpu_cores is None:
        """
        Default training -- no XLA
        """
        train(model, tokenizer, train_dataset, test_dataset, output_dir, num_steps, **args)
    else:
        """
        XLA training

        1. Save the datasets
        2. Wrap the model
        """
        import torch_xla.distributed.xla_multiprocessing as xmp


        if not dataset_path or (not on_the_fly):
            print("Saving datasets...")
            dataset_path = "/tmp/dataset_finetune_lm"
            dataset.save_to_disk(dataset_path)

        # print("Wrapping model...")
        # WRAPPED_MODEL = xmp.MpModelWrapper(model)


        print("Spawning")
        xmp.spawn(
            _mp_fn, args=(model_name, dataset_path, output_dir, num_steps, num_eval_batches, args),
            nprocs=num_tpu_cores, start_method="spawn"
        )

# def _mp_fn(*args, **kwargs):
#     return fire.Fire(finetune_lm)

if __name__ == '__main__':
    fire.Fire(finetune_lm)