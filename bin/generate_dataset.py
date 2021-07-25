import os
from glob import glob
import datasets
import random
import fire
from finetune_vs_scratch.model import load_tokenizer
from datasets import load_dataset, load_from_disk

random.seed(2021)

def generate_dataset(input_path, output_path, num_files=6):
    """
    Generate train and test split
    """
    tweet_files = random.sample(
        glob(os.path.join(input_path, "*.txt")),
        num_files
    )

    train_files, test_files = tweet_files[:-1], tweet_files[-1:]

    dataset = load_dataset("text", data_files={"train": train_files, "test": test_files})

    print("Saving")
    dataset.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")

if __name__ == '__main__':
    fire.Fire(generate_dataset)