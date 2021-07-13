from .preprocessing import special_tokens
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_name, num_labels, device, add_tokens=special_tokens):
    """
    Load model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=num_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_length = 128

    model = model.to(device)
    model.train()

    vocab = tokenizer.get_vocab()
    new_tokens_to_add = [tok for tok in add_tokens if tok not in vocab]

    if new_tokens_to_add:
        tokenizer.add_tokens(new_tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
