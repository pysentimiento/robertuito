import torch
import fire
import re
import tempfile
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from pysentimiento.tass import id2label as id2labeltass, label2id as label2idtass
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento.metrics import compute_metrics as compute_sentiment_metrics
from datasets import Dataset, Value, ClassLabel, Features

preprocess_args = {
    "user_token": "@usuario",
    "url_token": "url",
    "hashtag_token": "hashtag",
    "emoji_wrapper": "emoji",
}

special_tokens = ["@usuario", "url", "hashtag", "emoji"]

def my_preprocess(tweet):
    """
    My preprocess
    """
    ret = preprocess_tweet(tweet, **preprocess_args)
    ret = re.sub("\n+", ". ", ret)
    ret = re.sub(r"\s+", " ", ret)
    return ret.strip()

def load_datasets(limit=None):
    """
    Load sentiment datasets
    """
    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=3, names=["neg", "neu", "pos"])
    })
    df = pd.read_csv("data/tass.csv")
    df["label"] = df["polarity"].apply(lambda x: label2idtass[x])
    columns = ["text", "lang", "label"]

    train_dataset = Dataset.from_pandas(df[df["split"] == "train"], features=features)
    dev_dataset = Dataset.from_pandas(df[df["split"] == "dev"], features=features)
    test_dataset = Dataset.from_pandas(df[df["split"] == "test"], features=features)

    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))

    return train_dataset, dev_dataset, test_dataset


def load_model_and_tokenizer(model_name, device):
    """
    Load model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=len(id2labeltass)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_length = 128

    model = model.to(device)
    model.train()

    vocab = tokenizer.get_vocab()
    new_tokens_to_add = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]

    """
    TODO: Perdoname Wilkinson, te he fallado

    Hay una interfaz diferente acá, no entiendo bien por qué
    """
    if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens_to_add})
    else:
        tokenizer.add_special_tokens(new_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def run_sentiment(model_name, device):
    print("Running sentiment experiments")

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    train_dataset, dev_dataset, test_dataset = load_datasets(limit=250)

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)


    epochs = 5
    batch_size = 8
    eval_batch_size = 4
    accumulation_steps = 4
    lr = 1e-3

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in dataset.features:
            columns.append('token_type_ids')
        dataset.set_format(type='torch', columns=columns)
        print(columns)
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)


    output_path = tempfile.mkdtemp()
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        warmup_ratio=0.1,
        learning_rate=lr,
        evaluation_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda x: compute_sentiment_metrics(x, id2label=id2labeltass),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    return test_results


def run_benchmark(model_name: str, times: int):
    """
    Run benchmark

    Arguments:

    model_name:
        Model name or path. If a model name, should be present at huggingface's model hub

    times:
        Number of times
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(("*"*80+'\n')*3)
    print(f"Running benchmark with {model_name}")
    print(f"Using {device}", "\n"*3)
    print(("*"*80+'\n')*3)

    for i in range(times):
        print(("="*80+'\n')*3)
        print(f"{i+1} iteration", "\n"*3)

        sent_results = run_sentiment(model_name, device)





if __name__ == '__main__':
    fire.Fire(run_benchmark)