import logging
from torch.utils.data import IterableDataset, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BatchProcessedDataset(IterableDataset):
    """
    A dataset which streams and process tweets from files
    """
    def __init__(self, files, tokenizer, batch_size=4096, limit=None, padding='max_length'):
        self.files = files
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.padding = padding
        self.limit = limit

    def __nextbatch(self, f):
        nb = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]
        return nb

    def __iter__(self):
        num_iter = 0
        for file_path in self.files:
            logger.info(f"Opening file {file_path}")
            with open(file_path) as f:
                next_batch = self.__nextbatch(f)
                while next_batch:
                    tokenized_batch = self.tokenizer(next_batch, padding=self.padding, truncation=True, return_special_tokens_mask=True)
                    for encoding in tokenized_batch.encodings:
                        if self.limit and num_iter >= self.limit:
                            return
                        yield {
                            "input_ids": encoding.ids,
                            "token_type_ids": encoding.type_ids,
                            "attention_mask": encoding.attention_mask,
                            "special_tokens_mask": encoding.special_tokens_mask
                        }
                        num_iter += 1
                    old_batch = next_batch
                    next_batch = self.__nextbatch(f)

                    if next_batch:
                        """
                        Check we are not stuck in the same place
                        """
                        assert next_batch[0] != old_batch[0]


class DummyDataset(IterableDataset):
    """
    Just for test purposes

    Dataset that returns always the same. Used to check out MXU utilization in TPU
    """
    def __init__(self, ret, length):
        self.ret = ret
        self.length = length

    def __iter__(self):
        for _ in range(self.length):
            yield self.ret


class DummyRandomAccessDataset(Dataset):
    """
    Just for test purposes

    Dataset that returns always the same. Used to check out MXU utilization in TPU
    """
    def __init__(self, ret, length):
        self.ret = ret
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.ret
