import torch
import itertools
import logging
import pickle
import ujson
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

    @property
    def lines(self):
        """
        Generator through all the lines
        """

        for file_path in itertools.cycle(self.files):
            logger.info(f"Opening file {file_path}")
            with open(file_path) as f:
                for line in f:
                    stripped_line = line.strip("\n")
                    if not stripped_line:
                        continue
                    yield stripped_line

    @property
    def batches(self):
        """
        Generator through all the batches
        """
        lines = self.lines

        while True:
            yield list(itertools.islice(lines, self.batch_size))


    @property
    def encoded_batches(self):
        """
        Generator through all encoded batches
        """

        for batch in self.batches:
            tokenized_batch = self.tokenizer(
                batch, padding=self.padding, truncation=True,
                return_special_tokens_mask=True
            )

            encoded_batch = [{
                "input_ids": encoding.ids,
                "token_type_ids": encoding.type_ids,
                "attention_mask": encoding.attention_mask,
                "special_tokens_mask": encoding.special_tokens_mask
            } for encoding in tokenized_batch.encodings ]

            yield encoded_batch

    def __iter__(self):
        """
        Iterate through samples
        """
        logger.debug("Getting iterator")
        num_iter = 0
        for encoded_batch in self.encoded_batches:
            for encoding in encoded_batch:

                if self.limit and num_iter >= self.limit:
                    return
                yield encoding
                num_iter += 1


class JSONDistributedBatchProcessedDataset(BatchProcessedDataset):
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
        if self.is_master:
            self._encoded_batches = super().encoded_batches

    @property
    def encoded_batches(self):
        """

        """
        desc = "Distributed next batch"

        while True:
            try:
                if not self.is_master:
                    # tell all replicas to wait
                    logger.debug(f"waiting for the master to complete tokenization")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        torch.distributed.barrier()

                    """
                    Out of rendezvous => read data
                    """
                    with open(self.cache_file, "r") as load_file:
                        logger.debug(f"Reading JSON from {self.cache_file}")
                        next_batch = ujson.load(load_file)
                else:
                    """
                    Master => Process data
                    """
                    logger.debug("Waiting next batch at master")
                    next_batch  = next(self._encoded_batches)
                    with open(self.cache_file, "w") as dump_file:
                        ujson.dump(next_batch, dump_file)
            except Exception as e:
                logger.info(f"Exception at JSON batch encoding")
                logger.info(e)
                raise e
            finally:
                if self.is_master:
                    # the wait is over
                    logger.debug(f"Master completed {desc}, releasing all replicas")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        torch.distributed.barrier()

            logger.debug(self.tokenizer.decode(next_batch[0]["input_ids"], skip_special_tokens=True))
            yield next_batch

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
        if self.is_master:
            self._encoded_batches = super().encoded_batches

    @property
    def encoded_batches(self):
        """

        """
        desc = "Distributed next batch"

        while True:
            try:
                if not self.is_master:
                    # tell all replicas to wait
                    logger.debug(f"waiting for the master to complete tokenization")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        torch.distributed.barrier()

                    with open(self.cache_file, "rb") as pickle_file:
                        logger.debug(f"Reading from {self.cache_file}")
                        next_batch = pickle.load(pickle_file)
                else:
                    """
                    Master => Process data
                    """
                    next_batch  = next(self._encoded_batches)
                    with open(self.cache_file, "wb") as pickle_file:
                        logger.debug(f"Saving to {self.cache_file} -- len {len(next_batch)}")
                        pickle.dump(next_batch, pickle_file)
                        pickle_file.flush()
            except Exception as e:
                logger.debug(f"EXCEPTION -- Es maestro? {self.is_master} -- {e}")
                raise e
            finally:
                if self.is_master:
                    # the wait is over
                    logger.debug(f"Master completed {desc}, releasing all replicas")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        torch.distributed.barrier()

            yield next_batch
