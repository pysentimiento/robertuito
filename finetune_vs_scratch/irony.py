"""
Run sentiment experiments
"""
import torch
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from .model import load_model_and_tokenizer
from .preprocessing import preprocess
from .experiments import run_experiment


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
sentiment_dir = os.path.join(data_dir, "irony")

id2label = {
    0: 'not ironic',
    1: 'ironic',
}

label2id = {v:k for k, v in id2label.items()}


def load_datasets(data_path=None, limit=None, random_state=20202021, preprocess_data=True, preprocess_args=None):
    """
    Load sentiment datasets
    """
    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=2)
    })
    data_path = data_path or os.path.join(sentiment_dir, "irosva_dataset.csv")
    df = pd.read_csv(data_path)
    df["label"] = df["is_ironic"]


    if preprocess_data:
        preprocess_fn = lambda x: preprocess(x, preprocess_args=preprocess_args)
        df["text"] = df["text"].apply(preprocess_fn)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df, dev_df = train_test_split(
        train_df, stratify=train_df["label"], random_state=random_state,
        test_size=0.25,
    )

    train_dataset = Dataset.from_pandas(train_df, features=features)
    dev_dataset = Dataset.from_pandas(dev_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)


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
    print("Running Irony Detection experiments")



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
