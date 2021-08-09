"""
Run sentiment experiments
"""
import torch
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from pysentimiento.tass import id2label, label2id
from .model import load_model_and_tokenizer
from .preprocessing import preprocess
from .experiments import run_experiment


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




def run(
    model_name, device, data_path=None, limit=None, epochs=5, batch_size=32, max_length=128,
    eval_batch_size=16, accumulation_steps=1, use_dynamic_padding=True, **kwargs):
    """
    Run sentiment analysis experiments
    """
    print("Running sentiment experiments")



    model, tokenizer = load_model_and_tokenizer(
        model_name,
        num_labels=len(label2id),
        device=device,
        max_length=max_length
    )
    train_dataset, dev_dataset, test_dataset = load_datasets(
        data_path=data_path, limit=limit
    )


    return run_experiment(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        epochs=epochs, batch_size=batch_size, accumulation_steps=accumulation_steps,
        use_dynamic_padding=use_dynamic_padding, **kwargs,
    )
