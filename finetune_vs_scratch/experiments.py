import os
import torch
import tempfile
from pysentimiento.metrics import compute_metrics
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding



class MultiLabelTrainer(Trainer):
    """
    Multilabel and class weighted trainer
    """
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


def run_experiment(model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
    epochs=5, batch_size=32, accumulation_steps=1, format_dataset=None, eval_batch_size=16, use_dynamic_padding=True, class_weight=None, group_by_length=True,
    **kwargs):
    """
    Run experiments experiments
    """
    padding = False if use_dynamic_padding else 'max_length'
    def tokenize(batch):
        return tokenizer(batch['text'], padding=padding, truncation=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    data_collator = None
    if use_dynamic_padding:
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    else:
        if not format_dataset:
            raise ValueError("Must provide format_dataset if not using dynamic padding")

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
        save_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        group_by_length=group_by_length,
        **kwargs,
    )

    if class_weight is not None:
        class_weight = class_weight.to(device)
        trainer = MultiLabelTrainer(
            class_weight=class_weight,
            model=model,
            args=training_args,
            compute_metrics=lambda x: compute_metrics(x, id2label=id2label),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
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
