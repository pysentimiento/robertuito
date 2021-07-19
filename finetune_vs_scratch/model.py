from .preprocessing import special_tokens
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_tokenizer(model_name, max_length, model=None, tokenizer_class=AutoTokenizer):
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.model_max_length = max_length
    vocab = tokenizer.get_vocab()
    new_tokens_to_add = [tok for tok in special_tokens if tok not in vocab]

    if new_tokens_to_add:
        tokenizer.add_tokens(new_tokens_to_add)
        if model:
            model.resize_token_embeddings(len(tokenizer))
    return tokenizer

def load_model_and_tokenizer(model_name, num_labels, device, add_tokens=special_tokens, max_length=128):
    """
    Load model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=num_labels
    )


    model = model.to(device)
    model.train()
    tokenizer = load_tokenizer(model_name, max_length, model=model)

    return model, tokenizer
