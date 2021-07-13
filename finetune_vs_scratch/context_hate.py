import os
import json
import pandas as pd
import pathlib
import tempfile
import torch
from pandarallel import pandarallel
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset, Value, ClassLabel, Features
from transformers import Trainer, TrainingArguments
from .preprocessing import preprocess
from .model import load_model_and_tokenizer

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

    if limit:
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))

    return train_dataset, dev_dataset, test_dataset



class MultiLabelTrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_category_metrics(pred):
    """
    Compute metrics for hatespeech category classifier
    """

    labels = pred.label_ids
    preds = torch.sigmoid(torch.Tensor(pred.predictions)).round()

    ret = {
    }
    """
    Calculo F1 por cada posición. Asumo que cada categoría está alineada correctamente en la i-ésima posición
    """
    f1s = []
    precs = []
    recalls = []
    for i, cat in enumerate(extended_hate_categories):
        cat_labels, cat_preds = labels[:, i], preds[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary'
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower()+"_f1"] = f1

    ret["mean_f1"] = torch.Tensor(f1s).mean()
    ret["mean_precision"] = torch.Tensor(precs).mean()
    ret["mean_recall"] = torch.Tensor(recalls).mean()
    return ret


def compute_extended_category_metrics(dataset, pred):
    """
    Add F1 for Task A
    """
    metrics = compute_category_metrics(pred)
    hate_true = dataset["HATEFUL"]
    hate_pred = ((pred.predictions[:, 1:] > 0).sum(axis=1) > 0).astype(int)

    prec, recall, f1, _ = precision_recall_fscore_support(hate_true, hate_pred, average="binary")

    metrics.update({
        "hate_precision": prec,
        "hate_recall": recall,
        "hate_f1": f1,
    })
    return metrics


def run(model_name, device, limit=None, epochs=5, batch_size=16, eval_batch_size=16,
    max_length=256):
    """
    Run Context Hate Experiments
    """
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=len(extended_hate_categories), device=device, max_length=max_length)


    train_dataset, dev_dataset, test_dataset = load_datasets(limit=limit)

    def tokenize(batch):
        return tokenizer(
            batch["text"], batch["title"],
            padding='max_length', truncation='longest_first')

    print("Tokenizing and formatting datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    def format_dataset(dataset):
        def get_category_labels(examples):
            return {'labels': torch.Tensor([examples[cat] for cat in extended_hate_categories])}
        dataset = dataset.map(get_category_labels)
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)

    labels = torch.Tensor([train_dataset[c] for c in extended_hate_categories]).T

    class_weight = (1 / (2 * labels.mean(0))).to(device)

    if limit:
        class_weight = (1.0 * (~class_weight.isinf()) * class_weight).nan_to_num()

    output_path = tempfile.mkdtemp()

    accumulation_steps = 32 // batch_size
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        gradient_accumulation_steps = accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        class_weight=class_weight,
        compute_metrics=lambda pred: compute_extended_category_metrics(dev_dataset, pred),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()


    test_results = trainer.evaluate(test_dataset)
    os.system(f"rm -Rf {output_path}")
    return test_results