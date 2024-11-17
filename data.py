
import random
from typing import Iterator

import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

CACHE_DIR = "tmp/cache"
DATASET = "PleIAs/common_corpus"
PAD_TOKEN = "<pad>"

class BufferedDataloader:
    """Creates an iterator that yields shuffled batches of tokenized text."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        sequence_length: int,
        buffer_n_sequences: int = 1000,
    ):
        if batch_size > buffer_n_sequences:
            raise ValueError("Batch size cannot be greater than buffer size")
        if buffer_n_sequences / batch_size < 2:
            print(
                f"Warning: Buffer size ({buffer_n_sequences}) is less than twice the batch size ({batch_size}). This may lead to poor shuffling."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token=PAD_TOKEN)

        self.batch_size = batch_size

        self.sequence_length = sequence_length
        self.buffer_n_sequences = buffer_n_sequences
        self.buffer_BS = torch.empty(buffer_n_sequences, sequence_length, dtype=torch.long)

        self.available_indices: set[int] = set()  # Indices of buffer slots filled with batches
        self.refill_indices: set[int] = set(range(buffer_n_sequences))  # Start by filling all positions

        self.dataset_iterator = self.create_dataset_iterator()

    def create_dataset_iterator(self) -> Iterator[torch.Tensor]:
        """Creates an iterator over **sequences** of tokens. Handles splitting examples into full sequences."""
        dataset = load_dataset(DATASET, streaming=True, cache_dir=CACHE_DIR)

        for example in dataset["train"]:
            tokens = torch.tensor(self.tokenizer(example["text"])["input_ids"])

            num_full_sequences = len(tokens) // self.sequence_length
            if num_full_sequences == 0:
                continue

            sequences_NS = tokens[: num_full_sequences * self.sequence_length].split(self.sequence_length)
            for sequence_S in sequences_NS:
                yield sequence_S

    def fill_buffer(self) -> None:
        """Fills empty buffer slots with new batches."""
        print(f"Filling buffer with {len(self.refill_indices)} sequences")
        refilled_indices = []
        for refill_idx, sequence_S in tqdm(zip(self.refill_indices, self.dataset_iterator)):
            self.buffer_BS[refill_idx] = sequence_S
            refilled_indices.append(refill_idx)

        self.available_indices.update(refilled_indices)
        self.refill_indices.difference_update(refilled_indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        """Returns the next batch from the buffer."""
        if len(self.refill_indices) >= self.buffer_n_sequences // 2:
            # Fill buffer until either full or the dataset is exhausted
            self.fill_buffer()

        if len(self.available_indices) < self.batch_size:
            # If there's not enough sequences in the buffer, the dataset is exhausted
            raise StopIteration()

        batch_indices = random.sample(list(self.available_indices), self.batch_size)
        batch = self.buffer_BS[batch_indices]
        self.available_indices.difference_update(batch_indices)
        self.refill_indices.update(batch_indices)
        return batch

