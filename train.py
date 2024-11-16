import random
from typing import Iterator

import torch
from datasets import load_dataset  # type: ignore
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

from crosscoder import AcausalCrosscoder
from llm import LLMWithCache

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DTYPE = torch.bfloat16


def train():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    batch_size = 64
    sequence_length = 128
    buffer_n_sequences = 1000
    crosscoder_hidden_dim = 32_768
    n_layers_to_encode = 4
    dec_init_norm = 0.08

    lambda_ = 0.0
    lambda_max = 5.0
    lambda_n_steps = 1000
    lambda_step = lambda_max / lambda_n_steps

    llm = LLMWithCache(model_name).to(device)
    crosscoder = AcausalCrosscoder(n_layers_to_encode, llm.hidden_size, crosscoder_hidden_dim, dec_init_norm).to(device)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=5e-5)

    for step, batch_NS in enumerate(BufferedDataloader(model_name, batch_size, sequence_length, buffer_n_sequences)):
        batch_NS = batch_NS.to(device)
        optimizer.zero_grad()
        hidden_states_NLD = get_hidden_state(llm, n_layers_to_encode, batch_NS)
        _, losses = crosscoder.forward_train(hidden_states_NLD)

        loss = losses.reconstruction_loss + lambda_ * losses.sparsity_loss

        if step < lambda_n_steps:
            lambda_ += lambda_step

        print(f"loss: {loss.item():.4f}")
        clip_grad_norm_(crosscoder.parameters(), 1.0)
        loss.backward()
        optimizer.step()


def get_hidden_state(llm: LLMWithCache, n_layers_to_encode: int, batch_NS: torch.Tensor) -> torch.Tensor:
    hidden_states_NLD = llm.get_residual_stream_NLD(batch_NS)[:, :n_layers_to_encode]
    return torch.tensor(hidden_states_NLD, dtype=DTYPE).to(device)


DATASET = "PleIAs/common_corpus"
CACHE_DIR = "tmp/cache"
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

        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token=PAD_TOKEN)

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


if __name__ == "__main__":
    train()
