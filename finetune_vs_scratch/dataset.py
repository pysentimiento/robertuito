from torch.utils.data import IterableDataset, Dataset

class BatchProcessedDataset(IterableDataset):
    def __init__(self, files, tokenizer, batch_size=4096, limit=-1):
        self.files = files
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.limit = limit

    def __nextbatch(self, f):
        return [x.strip("\n") for _, x in zip(range(self.batch_size), f)]

    def __iter__(self):
        num_iter = 0
        for file_path in self.files:
            with open(file_path) as f:
                next_batch = self.__nextbatch(f)
                while next_batch:
                    tokenized_batch = self.tokenizer(next_batch, padding='max_length', truncation=True, return_special_tokens_mask=True)
                    for encoding in tokenized_batch.encodings:
                        if num_iter == self.limit:
                            return
                        yield {
                            "input_ids": encoding.ids,
                            "token_type_ids": encoding.type_ids,
                            "attention_mask": encoding.attention_mask,
                            "special_tokens_mask": encoding.special_tokens_mask
                        }
                        num_iter += 1
                    next_batch = self.__nextbatch(f)


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
