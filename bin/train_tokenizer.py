import os
import glob
import fire
from finetune_vs_scratch.preprocessing import special_tokens
from tokenizers import BertWordPieceTokenizer

def train_tokenizer(
    train_path: str, output_path: str, clean_text: bool = True, strip_accents: bool =False,
    lowercase: bool = False, vocab_size: int=40_000, min_frequency: int = 2, limit_alphabet:int = 600,
        ):
    """
    Train tokenizer

    Arguments:
    clean_text (bool, optional, defaults to True) – Whether to clean the text, by removing any control  characters and replacing all whitespaces by the classic one.

    strip_accents (bool, optional) – Whether to strip all accents. If this option is not specified (ie == None), then it will be determined by the value for lowercase (as in the original Bert).

    lowercase (bool, optional, defaults to True) – Whether to lowercase.

    """
    tweet_files = glob.glob(os.path.join(train_path, "*.txt"))

    print(f"Found {len(tweet_files)} files in {train_path}")

    tokenizer = BertWordPieceTokenizer(
        clean_text=clean_text,
        handle_chinese_chars=True,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )

    tokenizer.add_tokens(special_tokens)
    print(tokenizer)
    print("Added: ", special_tokens)
    print("Training...")

    tokenizer.train(
        tweet_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,

        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=limit_alphabet,
        wordpieces_prefix="##",
    )

    tokenizer.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    fire.Fire(train_tokenizer)
