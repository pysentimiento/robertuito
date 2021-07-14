"""
Run sentiment experiments
"""
import os
import pathlib
import tempfile
import pandas as pd
from .model import load_model_and_tokenizer
from .preprocessing import preprocess
from transformers import Trainer, TrainingArguments
from datasets import Dataset, Value, ClassLabel, Features
from pysentimiento.tass import id2label, label2id
from pysentimiento.metrics import compute_metrics as compute_sentiment_metrics


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
sentiment_dir = os.path.join(data_dir, "sentiment")


def load_datasets(data_path=None, limit=None):
    """
    Load sentiment datasets
    """
    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=3, names=["neg", "neu", "pos"])
    })
    data_path = data_path or os.path.join(sentiment_dir, "tass.csv")
    df = pd.read_csv(data_path)
    df["label"] = df["polarity"].apply(lambda x: label2id[x])
    df["text"] = df["text"].apply(lambda x: preprocess(x))
    columns = ["text", "lang", "label"]

    train_dataset = Dataset.from_pandas(df[df["split"] == "train"], features=features)
    dev_dataset = Dataset.from_pandas(df[df["split"] == "dev"], features=features)
    test_dataset = Dataset.from_pandas(df[df["split"] == "test"], features=features)


    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))
        dev_dataset = dev_dataset.select(range(min(limit, len(dev_dataset))))
        test_dataset = test_dataset.select(range(min(limit, len(test_dataset))))


    return train_dataset, dev_dataset, test_dataset






def run(model_name, device, data_path=None, limit=None, epochs=5, batch_size=32, eval_batch_size=32):
    """
    Run sentiment analysis experiments
    """
    print("Running sentiment experiments")

    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=len(label2id), device=device)
    train_dataset, dev_dataset, test_dataset = load_datasets(data_path=data_path, limit=limit)

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    accumulation_steps = 32 // batch_size

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in dataset.features:
            columns.append('token_type_ids')
        dataset.set_format(type='torch', columns=columns)
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
        compute_metrics=lambda x: compute_sentiment_metrics(x, id2label=id2label),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

    test_results = trainer.evaluate(test_dataset)

    os.system(f"rm -Rf {output_path}")
    return test_results
