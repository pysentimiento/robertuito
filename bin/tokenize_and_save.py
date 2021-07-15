import datasets
import random
import fire
import torch
from transformers import BertTokenizerFast
from datasets import load_dataset, load_from_disk

random.seed(2021)

def tokenize_and_save(input_path, output_path, limit=25600000, model_name = 'dccuchile/bert-base-spanish-wwm-cased', max_length=128):
    """
    Tokenize and save arrow dataset
    """
    print("Loading dataset")
    dataset = load_from_disk(input_path)
    print(dataset)
    if len(dataset) > limit:
        print("Subsampling")
        idx = random.sample(list(range(len(dataset))), limit)
        dataset = dataset.select(idx)

    print(f"Loading {model_name} tokenizer -- {max_length} max len")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.model_max_length = max_length

    print("Tokenizing")
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_special_tokens_mask=True)

    dataset = dataset.map(tokenize, batch_size=2048, batched=True)

    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    fire.Fire(tokenize_and_save)