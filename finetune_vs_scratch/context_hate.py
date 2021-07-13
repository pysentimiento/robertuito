from .model import load_model_and_tokenizer
import os
import json
import pandas as pd
import pathlib
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from datasets import Dataset, Value, ClassLabel, Features
from .preprocessing import preprocess

pandarallel.initialize()

"""
These are the posible categories of hatespeech
"""
hate_categories = [
    "WOMEN", # Against women
    "LGBTI", # Against LGBTI
    "RACISM", # Racist
    "CLASS",  # Classist
    "POLITICS", # Because of politics
    "DISABLED", # Against disabled
    "APPEARANCE",  # Against people because their appearance
    "CRIMINAL", # Against criminals
]

"""
Categories + CALLS (call for action)
"""
extended_hate_categories = ["CALLS"] + hate_categories


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data", "context_hate")

_train_path = os.path.join(data_dir, "train.json")
_test_path = os.path.join(data_dir, "test.json")


def serialize(article, comment):
    """
    Serializes article and comment
    """
    ret = comment.copy()
    ret["title"] = article["title"]

    return ret



def load_datasets(train_path=None, test_path=None, limit=None):
    """
    Load and return datasets

    Returns
    -------

        train_dataset, dev_dataset, test_datasets: datasets.Dataset
    """
    test_path = test_path or _test_path
    train_path = train_path or _train_path

    with open(train_path) as f:
        train_articles = json.load(f)

    with open(test_path) as f:
        test_articles = json.load(f)


    train_comments = [
        serialize(article, comment)
        for article in train_articles for comment in article["comments"]]
    test_comments = [
        serialize(article, comment)
        for article in test_articles for comment in article["comments"]]

    if limit:
        train_comments = train_comments[:limit]
        test_comments = test_comments[:limit]
    train_df = pd.DataFrame(train_comments)
    test_df = pd.DataFrame(test_comments)

    train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=20212021)

    """
    Apply preprocessing: convert usernames to "usuario" and urls to URL
    """

    for df in [train_df, dev_df, test_df]:
        df["text"] = df["text"].parallel_apply(preprocess)
        df["title"] = df["title"].parallel_apply(preprocess)

    features = Features({
        'id': Value('uint64'),
        'title': Value('string'),
        'text': Value('string'),
        'HATEFUL': ClassLabel(num_classes=2, names=["Not Hateful", "Hateful"])
    })


    for cat in extended_hate_categories:
        """
        Set for WOMEN, LGBTI...and also for CALLS
        """
        features[cat] = ClassLabel(num_classes=2, names=["NO", "YES"])

    columns = list(features.keys())

    train_dataset = Dataset.from_pandas(train_df[columns], features=features)
    dev_dataset = Dataset.from_pandas(dev_df[columns], features=features)
    test_dataset = Dataset.from_pandas(test_df[columns], features=features)

    return train_dataset, dev_dataset, test_dataset



def run(model_name, device, limit=None, epochs=5, batch_size=32, eval_batch_size=32):
    """
    Run Context Hate Experiments
    """
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=len(extended_hate_categories), device=device)


    train_dataset, dev_dataset, test_dataset = load_datasets(limit=limit)
