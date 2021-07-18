
"""
Run sentiment experiments
"""
import torch
import pandas as pd
import os
import tempfile
import pathlib
from .preprocessing import preprocess, special_tokens
from .model import load_model_and_tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from pysentimiento.emotion.datasets import id2label, label2id
from pysentimiento.metrics import compute_metrics
from.preprocessing import preprocess, special_tokens

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


def load_datasets(train_path=None, test_path=None, limit=None,random_state=2021):
    """
    Load emotion recognition datasets
    """

    train_path = train_path or os.path.join(emotion_dir, "train_es.csv")
    test_path = test_path or os.path.join(emotion_dir, "test_es.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, dev_df = train_test_split(train_df, stratify=train_df["label"], random_state=random_state)


    for df in [train_df, dev_df, test_df]:
        for label, idx in label2id.items():
            df.loc[df["label"] == label, "label"] = idx
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].apply(preprocess)


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




class MultiLabelTrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weight)
        num_labels = self.model.config.num_labels
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def run(model_name, device, train_path=None, test_path=None, limit=None, epochs=5, batch_size=32, eval_batch_size=16,
        use_dynamic_padding=True, **kwargs):
    """
    Run sentiment analysis experiments
    """
    print("Running sentiment experiments")



    model, tokenizer = load_model_and_tokenizer(model_name, num_labels=len(label2id), device=device)
    train_dataset, dev_dataset, test_dataset = load_datasets(train_path=train_path, test_path=test_path, limit=limit)


    padding = False if use_dynamic_padding else 'max_length'
    def tokenize(batch):
        return tokenizer(batch['text'], padding=padding, truncation=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weight = torch.Tensor(
        compute_class_weight('balanced', list(id2label), y=train_dataset["label"])
    ).to(device)

    accumulation_steps = 32 // batch_size

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    data_collator = None
    if use_dynamic_padding:
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    else:
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
        **kwargs,
    )


    trainer = MultiLabelTrainer(
        class_weight=class_weight,
        model=model,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, id2label=id2label),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )


    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    os.system(f"rm -Rf {output_path}")
    return test_results
