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
    AutoModelForMaskedLM, AutoTokenizer, AutoConfig,
)
from finetune_vs_scratch.model import load_tokenizer
from torch.utils.data import IterableDataset

def tokenize(tokenizer, batch, padding='max_length'):
    return tokenizer(batch['text'], padding=padding, truncation=True, return_special_tokens_mask=True)

class BatchProcessedDataset(IterableDataset):
    def __init__(self, files, tokenizer, batch_size=4096, limit=-1):
        self.files = files
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.limit = limit

    def __iter__(self):
        num_iter = 0
        for file_path in self.files:
            with open(file_path) as f:

                next_batch = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]

                while next_batch:
                    tokenized_batch = self.tokenizer(next_batch, padding='max_length', truncation=True, return_special_tokens_mask=True)
                    for encoding in tokenized_batch.encodings:
                        if num_iter == self.limit:
                            return
                        yield {
                            "input_ids": encoding.ids,
                            "token_type_ids": encoding.type_ids,
                            "attention_mask": encoding.attention_mask,
                            "special_tokens_mask": encoding.special_tokens_mask
                        }
                        num_iter += 1
                    next_batch = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]


def run_mlm(
    output_dir:str, num_steps:int, input_dir, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    seed=2021, max_eval_steps=100, limit=None, eval_steps=200, save_steps=1000,
    padding='max_length', on_the_fly=False, tok_batch_size=1024*16,
    resume_from_checkpoint=None, finetune=False, logging_steps=100,
    per_device_batch_size=32, accumulation_steps=32,
    weight_decay=0.01, warmup_ratio=0.06, learning_rate=5e-4,
    adam_beta1=0.9, adam_beta2=0.98, max_grad_norm=0, ignore_data_skip=True,
    tpu_num_cores=None,
):
    """
    Run MLM

    Arguments:

    output_dir: str
        Where to save

    num_steps: int
        Number of steps to perform

    input_dir: str (default None)
        Where to look for tweet files

    """
    print(limit)

    print("Loading model")
    if finetune:
        """
        If finetune => Load model and use special `load_tokenizer` function
        """
        print("Finetuning pretrained model")
        model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        tokenizer = load_tokenizer(model_name, 128, model=model)
    else:
        """
        Pretraining from scratch
        """
        print(f"Pretraining from scratch -- {model_name} ")
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = 128
        model = AutoModelForMaskedLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))
        print(model)

    print(f"Padding {padding}")


    print("Loading datasets")

    tweet_files = glob(os.path.join(input_dir, "*.txt"))
    random.shuffle(tweet_files)
    print(f"Selecting {len(tweet_files)} files")
    print(f"First: {tweet_files[:3]}")
    train_files, test_files = tweet_files[:-1], tweet_files[-1:]

    train_dataset = BatchProcessedDataset(
        train_files, tokenizer, tok_batch_size)
    test_dataset = BatchProcessedDataset(
        test_files, tokenizer, tok_batch_size, limit=2048 * max_eval_steps
    )
    random.seed(seed)



    args = {
        "eval_steps":eval_steps,
        "save_steps":save_steps,
        "logging_steps": logging_steps,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "max_grad_norm": max_grad_norm,
        "ignore_data_skip": ignore_data_skip or on_the_fly,
        "adam_epsilon": 1e-6,
        "tpu_num_cores": tpu_num_cores,
    }

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



    # if on_the_fly:
    #     with training_args.main_process_first(desc="dataset map tokenization"):
    #         print("On the fly tokenization")
    #         train_dataset.set_transform(lambda x: tokenize(tokenizer, x, padding))
    #         test_dataset.set_transform(lambda x: tokenize(tokenizer, x, padding))
    # else:
    #     print("Tokenization preprocessing")
    #     print(len(train_dataset))
    #     print(len(test_dataset))
    #     with training_args.main_process_first(desc="dataset map tokenization"):
    #         print("Tokenizing")
    #         train_dataset = train_dataset.map(lambda x: tokenize(tokenizer, x, padding), batched=True, batch_size=batch_size, num_proc=num_proc)
    #         test_dataset = test_dataset.map(lambda x: tokenize(tokenizer, x, padding), batched=True, batch_size=batch_size, num_proc=num_proc)
    #         train_dataset = train_dataset.remove_columns(["text"])
    #         test_dataset = test_dataset.remove_columns(["text"])
    with training_args.main_process_first(desc="Checking lengths"):
        print("Sanity check")
        print(f"@usuario => {tokenizer.encode('@usuario')}")
        text = "esta es una PRUEBA EN MAYÚSCULAS Y CON TILDES @usuario @usuario"
        print(f"{text} ==> {tokenizer.decode(tokenizer.encode(text))}")
        print(training_args)
        print("Checking lengths")
        train_lengths = {len(ex["input_ids"]) for ex, _ in zip(train_dataset, range(20_000))}
        test_lengths = {len(ex["input_ids"]) for ex, _ in zip(test_dataset, range(20_000))}
        assert len(train_lengths) == 1
        assert len(test_lengths) == 1

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def _mp_fn(*args, **kwargs):
    return fire.Fire(run_mlm)

if __name__ == '__main__':
    fire.Fire(run_mlm)