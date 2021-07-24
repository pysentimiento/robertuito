import os
import fire
import datasets
from glob import glob
from datasets import load_dataset, Features, Value, load_from_disk
import torch
from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from torch.utils.data.dataloader import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer
from finetune_vs_scratch.preprocessing import special_tokens
from finetune_vs_scratch.model import load_model_and_tokenizer, load_tokenizer
import torch_xla.core.xla_model as xm

def train(model, tokenizer, train_dataset, test_dataset, output_dir, num_steps, **kwargs):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


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

def _mp_fn(index, WRAPPED_MODEL, tokenizer, train_path, test_path, output_dir, num_steps, training_args):
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)

    print("Reloading datasets at ", index)
    train_dataset = load_from_disk(train_path)
    test_dataset = load_from_disk(test_path)

    return train(model, tokenizer, train_dataset, test_dataset, output_dir, num_steps, **training_args)

def finetune_lm(
    input_dir, output_dir, num_steps, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    batch_size=2048, num_eval_batches=20, deepspeed=None, limit=None, eval_steps=200, save_steps=1000,
    per_device_batch_size=32, accumulation_steps=32, warmup_ratio=0.06, weight_decay=0.01, learning_rate=5e-4, on_the_fly=False,
    num_tpu_cores=None,
):
    """
    Finetune LM
    """

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
    padding = 'max_length' if num_tpu_cores else False

    print(f"Padding {padding}")

    print("Sanity check")
    print(f"@usuario => {tokenizer.encode('@usuario')}")
    text = train_dataset[0]["text"]
    print(f"{text} ==> {tokenizer.decode(tokenizer.encode(text))}")

    print("Tokenizing")

    """
    If TPU => pad to max length

    See https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#known-performance-caveats
    """
    def tokenize(batch):
        return tokenizer(batch['text'], padding=padding, truncation=True, return_special_tokens_mask=True)

    if on_the_fly:
        train_dataset.set_transform(tokenize)
        test_dataset.set_transform(tokenize)
    else:
        train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=24)
        test_dataset = test_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=24)
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

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

        train_path = "/tmp/train"
        test_path = "/tmp/test"
        train_dataset.save_to_disk(train_path)
        test_dataset.save_to_disk(test_path)

        WRAPPED_MODEL = xmp.MpModelWrapper(model)



        xmp.spawn(
            _mp_fn, args=(WRAPPED_MODEL, tokenizer, train_path, test_path, output_dir, num_steps, args),
            nprocs=num_tpu_cores, start_method="fork"
        )

# def _mp_fn(*args, **kwargs):
#     return fire.Fire(finetune_lm)

if __name__ == '__main__':
    fire.Fire(finetune_lm)