import os
import glob
import fire
from finetune_vs_scratch.preprocessing import special_tokens
from finetune_vs_scratch.tokenizer import tokenizer_special_tokens
from tokenizers import SentencePieceBPETokenizer, normalizers
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import RobertaProcessing

def train_tokenizer(
    train_path: str, output_path: str, strip_accents: bool =False,
    lowercase: bool = False, vocab_size: int=30_000, min_frequency: int = 10, limit_alphabet:int = 400, num_files=40,
        ):
    """
    Train tokenizer

    Arguments:
    clean_text (bool, optional, defaults to True) – Whether to clean the text, by removing any control  characters and replacing all whitespaces by the classic one.

    strip_accents (bool, optional) – Whether to strip all accents. If this option is not specified (ie == None), then it will be determined by the value for lowercase (as in the original Bert).

    lowercase (bool, optional, defaults to True) – Whether to lowercase.

    """
    tweet_files = sorted(
        glob.glob(os.path.join(train_path, "*.txt"))
    )[:num_files]

    print(f"Found {len(tweet_files)} files in {train_path}")

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.add_special_tokens(tokenizer_special_tokens)

    print(tokenizer)
    print("Training...")
    print(f"Lowercase: {lowercase}")
    print(f"Strip accents: {strip_accents}")

    tokenizer_normalizers = [
        normalizers.NFKC(),
        normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
    ]


    tokenizer.post_processor = RobertaProcessing(
        cls=("<s>", tokenizer.token_to_id("<s>")),
        sep=("</s>", tokenizer.token_to_id("</s>")),
    )

    print(tokenizer_normalizers)
    tokenizer.normalizer = normalizers.Sequence(tokenizer_normalizers)
    print(tokenizer.normalizer)

    tokenizer.train(
        tweet_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,

        special_tokens=tokenizer_special_tokens + special_tokens,
        limit_alphabet=limit_alphabet,
    )
    inv_vocab = {v:k for k, v in tokenizer.get_vocab().items()}
    print(f"First tokens: {[inv_vocab[i] for i in range(20)]}")

    alphabet = sorted(list({a for x in tokenizer.get_vocab() for a in x}))
    print("Alphabet = ", " ".join(alphabet))



    transformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    text = "@usuario ESTO es una prueba esdrújula PAPÁ"

    print(f"Without tokenizing: {text}")
    decoded = transformer_tokenizer.decode(
       transformer_tokenizer(text)["input_ids"]
    )
    print(f"Processed: {decoded}")


    text = ["@usuario dos oraciones", "segunda ORACIÓN"]

    print(f"Without tokenizing: {text}")
    decoded = transformer_tokenizer.decode(
       transformer_tokenizer(*text)["input_ids"]
    )
    print(f"Processed: {decoded}")


    transformer_tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(train_tokenizer)
