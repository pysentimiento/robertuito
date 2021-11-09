
"""
Run emotion experiments
"""
import torch
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from pysentimiento.emotion.datasets import id2label, label2id
from .model import load_model_and_tokenizer
from .preprocessing import preprocess
from .experiments import run_experiment


id2label = {
    0: 'others',
    1: 'joy',
    2: 'sadness',
    3: 'anger',
    4: 'surprise',
    5: 'disgust',
    6: 'fear',
}

label2id = {v:k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
emotion_dir = os.path.join(data_dir, "emotion")


def load_datasets(train_path=None, test_path=None, limit=None,random_state=2021, preprocess_data=True, preprocess_args=None):
    """
    Load emotion recognition datasets
    """

    train_path = train_path or os.path.join(emotion_dir, "train_es.csv")
    test_path = test_path or os.path.join(emotion_dir, "test_es.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, dev_df = train_test_split(train_df, stratify=train_df["label"], random_state=random_state)


    preprocess_fn = lambda x: preprocess(x, preprocess_args=preprocess_args)
    for df in [train_df, dev_df, test_df]:
        for label, idx in label2id.items():
            df.loc[df["label"] == label, "label"] = idx
        df["label"] = df["label"].astype(int)

        if preprocess_data:
            df["text"] = df["text"].apply(preprocess_fn)


    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=len(id2label), names=[id2label[k] for k in sorted(id2label.keys())])
    })

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
    model_name, device, train_path=None, test_path=None, limit=None, epochs=5, batch_size=32,
    eval_batch_size=16, max_length=128, accumulation_steps=1, use_dynamic_padding=True, **kwargs):
    """
    Run emotion experiments
    """
    print("Running emotion experiments")



    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=len(label2id), device=device, max_length=max_length)
    train_dataset, dev_dataset, test_dataset = load_datasets(train_path=train_path, test_path=test_path, limit=limit)

    class_weight = torch.Tensor(
        compute_class_weight('balanced', list(id2label), y=train_dataset["label"])
    )

    return run_experiment(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        class_weight=class_weight, epochs=epochs, batch_size=batch_size, accumulation_steps=accumulation_steps, use_dynamic_padding=use_dynamic_padding,
        **kwargs,
    )