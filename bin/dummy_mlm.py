"""


See: https://github.com/huggingface/transformers/pull/10255

"""

import fire
from torch.utils.data import IterableDataset

from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    AutoModelForMaskedLM, AutoTokenizer, AutoConfig,
)


class DummyDataset(IterableDataset):
    """
    Just for test purposes

    Dataset that returns always the same. Used to check out MXU utilization in TPU
    """
    def __init__(self, ret, length):
        self.ret = ret
        self.length = length

    def __iter__(self):
        for _ in range(self.length):
            yield self.ret

def run_mlm(
    output_dir:str, num_steps:int, model_name = 'dccuchile/bert-base-spanish-wwm-uncased',
    limit=None, eval_steps=200, save_steps=1000,
    padding='max_length',resume_from_checkpoint=None, logging_steps=500,
    per_device_batch_size=32, accumulation_steps=32,
    weight_decay=0.01, warmup_ratio=0.06, learning_rate=5e-4,
    adam_beta1=0.9, adam_beta2=0.98, max_grad_norm=0,
    tpu_num_cores=None
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
    """
    Pretraining from scratch
    """
    print(f"Pretraining from scratch -- {model_name} ")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 128
    model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    #print(model)

    print(f"Padding {padding}")

    train_dataset = DummyDataset(
        tokenizer("Esto es una prueba @usuario", padding=padding, truncation=True, return_special_tokens_mask=True),
        50_000_000
    )

    test_dataset = train_dataset


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
        "ignore_data_skip": True,
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