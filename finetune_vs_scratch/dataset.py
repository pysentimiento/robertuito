import torch
import logging
import pickle
import json
from torch.utils.data import IterableDataset, Dataset
from transformers.file_utils import (
    is_torch_tpu_available,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

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

    def _next_batch(self, f):
        next_batch = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]

        tokenized_batch = self.tokenizer(next_batch, padding=self.padding, truncation=True, return_special_tokens_mask=True)

        encoded_batch = [{
            "input_ids": encoding.ids,
            "token_type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask,
            "special_tokens_mask": encoding.special_tokens_mask
        } for encoding in tokenized_batch.encodings ]

        return encoded_batch

    def __iter__(self):
        num_iter = 0
        for file_path in self.files:
            logger.info(f"Opening file {file_path}")
            with open(file_path) as f:
                next_batch = self._next_batch(f)
                while next_batch:
                    for encoding in next_batch:
                        if self.limit and num_iter >= self.limit:
                            return
                        yield encoding
                        num_iter += 1
                    next_batch = self._next_batch(f)

class DistributedBatchProcessedDataset(BatchProcessedDataset):
    """
    A BatchProcessedDataset that takes care of distributed environment and tokenizes stuff just once
    """

    def __init__(self, files, tokenizer, is_master, cache_file, **kwargs):
        """
        Constructor


        is_master: bool
            Is this the master process? Master process tokenizes and saves each batch

        cache_file: str (a path)
            Where to save the cached tokenization
        """
        self.training_args = is_master
        self.cache_file = cache_file
        self.is_master = is_master
        super().__init__(files, tokenizer, **kwargs)

    def _next_batch(self, f):
        desc = "Distributed next batch"
        try:
            if not self.is_master:
                # tell all replicas to wait
                logger.info(f"waiting for the master to complete tokenization")
                if is_torch_tpu_available():
                    xm.rendezvous(desc)
                else:
                    torch.distributed.barrier()
            else:
                """
                Master => Process data
                """
                next_batch  = super()._next_batch(f)
                with open(self.cache_file, "wb") as pickle_file:
                    logger.info(f"Saving to {self.cache_file}")
                    pickle.dump(next_batch, pickle_file)
        finally:
            if self.is_master:
                # the wait is over
                logger.info(f"{self.process_index}: Master completed {desc}, releasing all replicas")
                if is_torch_tpu_available():
                    xm.rendezvous(desc)
                else:
                    torch.distributed.barrier()
            else:
                with open(self.cache_file, "rb") as pickle_file:
                    logger.info(f"Reading from {self.cache_file}")
                    next_batch = pickle.load(pickle_file)

        return next_batch


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
