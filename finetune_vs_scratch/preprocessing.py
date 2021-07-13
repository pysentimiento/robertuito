import re
from pysentimiento.preprocessing import preprocess_tweet

preprocess_args = {
    "user_token": "@usuario",
    "url_token": "url",
    "hashtag_token": "hashtag",
    "emoji_wrapper": "emoji",
}

special_tokens = ["@usuario", "url", "hashtag", "emoji"]

def preprocess(tweet):
    """
    My preprocess
    """
    ret = preprocess_tweet(tweet, **preprocess_args)
    ret = re.sub("\n+", ". ", ret)
    ret = re.sub(r"\s+", " ", ret)
    return ret.strip()