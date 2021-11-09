
"""
Run hatEval experiments
"""
import pandas as pd
import os
import pathlib
from .preprocessing import preprocess
from .model import load_model_and_tokenizer
from datasets import Dataset, Value, ClassLabel, Features
from pysentimiento.emotion.datasets import id2label, label2id
from .preprocessing import preprocess, special_tokens
from .experiments import run_experiment

id2label = {
    0: 'ok',
    1: 'hateful',
}

label2id = {v:k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data", "hate")


def load_datasets(train_path=None, dev_path=None, test_path=None, limit=None, preprocess_data=True, preprocess_args=None, random_state=2021):
    """
    Load emotion recognition datasets
    """

    train_path = train_path or os.path.join(data_dir, "hateval2019_es_train.csv")
    dev_path = dev_path or os.path.join(data_dir, "hateval2019_es_dev.csv")
    test_path = test_path or os.path.join(data_dir, "hateval2019_es_test.csv")


    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)


    preprocess_fn = lambda x: preprocess(x, preprocess_args=preprocess_args)
    for df in [train_df, dev_df, test_df]:
        df["label"] = df["HS"].astype(int)
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
    model_name, device, limit=None, epochs=5, batch_size=32, max_length=128,
    eval_batch_size=16, accumulation_steps=1, use_dynamic_padding=True, **kwargs):
    """
    Run sentiment analysis experiments
    """
    print("Running hatEval experiments")


    model, tokenizer = load_model_and_tokenizer(
        model_name,
        num_labels=len(label2id),
        device=device,
        max_length=max_length
    )

    train_dataset, dev_dataset, test_dataset = load_datasets(
        limit=limit
    )


    return run_experiment(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        epochs=epochs, batch_size=batch_size, accumulation_steps=accumulation_steps,
        use_dynamic_padding=use_dynamic_padding, **kwargs,
    )
