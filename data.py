from functools import cached_property, partial
from typing import Generator, Iterator, TypeVar

from einops import rearrange
import torch
import random
from transformer_lens import HookedTransformer  # type: ignore

from git import Optional
import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

CACHE_DIR = "tmp/cache"
DATASET = "PleIAs/common_corpus"
PAD_TOKEN = "<pad>"

"""
dataset:
 [
    "asdfasdf",
    "asdff",
    "asdfdf",
    ...
 ]

tokens:
 [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 0, 0],
    ...
 ]
"""


def shuffle_buffer_iterator(
    iterator: Iterator[torch.Tensor],
    buffer_size: int,
    batch_size: int,
    item_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> Iterator[torch.Tensor]:
    if batch_size > buffer_size // 2:
        raise ValueError("Batch size cannot be greater than half the buffer size")

    buffer = torch.empty(buffer_size, *item_shape, dtype=dtype, device=device)
    available_indices = set()
    stale_indices = set(range(buffer_size))
    while True:
        # fill buffer
        for refill_idx, sequence_S in zip(stale_indices, iterator):
            buffer[refill_idx] = sequence_S
            available_indices.add(refill_idx)
            stale_indices.remove(refill_idx)

        # yield batches
        while len(available_indices) >= buffer_size // 2:
            batch_indices = random.sample(list(available_indices), batch_size)
            batch_BS = buffer[batch_indices]
            available_indices.difference_update(batch_indices)
            stale_indices.update(batch_indices)
            yield batch_BS


class BufferedActivationLoader:
    def __init__(
        self,
        layer_indices_to_encode: list[int],
        model_name: str,
        activation_batch_size: int,
        buffer_size: int,
        sequence_length: int,
        device: str,
        model_dtype: torch.dtype,
        sequence_length: int,
    ):
        # if batch_size > buffer_n_sequences:
        #     raise ValueError("Batch size cannot be greater than buffer size")
        # if buffer_n_sequences / batch_size < 2:
        #     print(
        #         f"Warning: Buffer size ({buffer_n_sequences}) is less than twice the batch size ({batch_size}). This may lead to poor shuffling."
        #     )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token=PAD_TOKEN)
        self._model: HookedTransformer = HookedTransformer.from_pretrained(model_name, cache_dir=CACHE_DIR, dtype=model_dtype).to(device)

        self._batch_size = activation_batch_size
        self.activation_shuffle_buffer = partial(
            shuffle_buffer_iterator,
            buffer_size=buffer_size,
            batch_size=activation_batch_size,
            item_shape=(sequence_length,),
            dtype=model_dtype,
            device=device,
        )
        self.layer_indices_to_encode = layer_indices_to_encode
        self._sequence_length = sequence_length

    @cached_property
    def _tokens_iterator_S(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(DATASET, streaming=True, cache_dir=CACHE_DIR)

        for example in dataset["train"]:
            tokens_BS = torch.tensor(self._tokenizer(example["text"])["input_ids"])
            num_full_sequences = len(tokens_BS) // self._sequence_length
            if num_full_sequences == 0:
                continue

            sequences_NS = tokens_BS[: num_full_sequences * self._sequence_length].split(self._sequence_length)
            for sequence_S in sequences_NS:
                yield sequence_S

    @cached_property
    def _activation_iterator(self) -> Iterator[torch.Tensor]:
        names = {f"blocks.{num}.hook_resid_post" for num in self.layer_indices_to_encode}
        def names_filter(name: str) -> bool:
            return name in names

        with torch.no_grad():
            for sequence_BS in batched(self._tokens_iterator_S, self._batch_size):
                _, cache = self._model.run_with_cache(sequence_BS, names_filter=names_filter)
                activations_NSD = []
                for layer_idx in self.layer_indices_to_encode:
                    activation_NSD = cache[("resid_post", layer_idx)]
                    activations_NSD.append(activation_NSD)
                activations_LNSD = torch.stack(activations_NSD, dim=0)
                activations_NsLD = rearrange(activations_LNSD, "l n s d -> (n s) l d")



            yield activations_BS

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        """Returns the next batch from the buffer."""
        if len(self._refill_indices) >= self._buffer_n_sequences // 2:
            # Fill buffer until either full or the dataset is exhausted
            self._fill_buffer()

        if len(self._available_indices) < self._batch_size:
            # If there's not enough sequences in the buffer, the dataset is exhausted
            raise StopIteration()

        batch_indices = random.sample(list(self._available_indices), self._batch_size)
        batch_BS = self._buffer_BS[batch_indices]
        self._available_indices.difference_update(batch_indices)
        self._refill_indices.update(batch_indices)
        return batch_BS

def batched(iterator: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []

# following code is copied / adapted from https://github.com/JBloomAUS/SAELens/blob/main/sae_lens/tokenization_and_batching.py


def _add_tokens_to_batch(
    batch: torch.Tensor | None,
    tokens: torch.Tensor,
    offset: int,
    context_size: int,
    # is_start_of_sequence: bool,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
) -> tuple[torch.Tensor, int]:
    if context_size < 10:
        raise ValueError("Context size must be at least 10, can't be bothered dealing with errors at this level")

    prefix_toks = []
    first_token = tokens[offset]
    # prepend the start of sequence token if needed
    if offset == 0 and begin_sequence_token_id is not None:
        begin_sequence_token_id_tensor = torch.tensor([begin_sequence_token_id], dtype=torch.long, device=tokens.device)
        if first_token != begin_sequence_token_id_tensor:
            prefix_toks.insert(0, begin_sequence_token_id_tensor)
            first_token = begin_sequence_token_id_tensor
    # We're at the start of a new batch
    if batch is None:
        # add the BOS token to the start if needed
        if begin_batch_token_id is not None:
            begin_batch_token_id_tensor = torch.tensor([begin_batch_token_id], dtype=torch.long, device=tokens.device)
            if first_token != begin_batch_token_id_tensor:
                prefix_toks.insert(0, begin_batch_token_id_tensor)
                first_token = begin_batch_token_id_tensor
        tokens_needed = max(context_size - len(prefix_toks), 0)
        tokens_part = tokens[offset : offset + tokens_needed]
        batch = torch.cat([*prefix_toks[:context_size], tokens_part])
        return batch, offset + tokens_needed
    # if we're concatting batches, add the separator token as needed
    if sequence_separator_token_id is not None:
        sequence_separator_token_id_tensor = torch.tensor(
            [sequence_separator_token_id], dtype=torch.long, device=tokens.device
        )
        if first_token != sequence_separator_token_id_tensor:
            prefix_toks.insert(0, sequence_separator_token_id_tensor)
            first_token = sequence_separator_token_id_tensor
    tokens_needed = max(context_size - batch.shape[0] - len(prefix_toks), 0)
    prefix_toks_needed = max(context_size - batch.shape[0], 0)
    batch = torch.concat(
        [
            batch,
            *prefix_toks[:prefix_toks_needed],
            tokens[offset : offset + tokens_needed],
        ]
    )
    return batch, offset + tokens_needed


@torch.no_grad()
def concat_and_batch_sequences(
    tokens_iterator: Iterator[torch.Tensor],
    context_size: int,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Generator to concat token sequences together from the tokens_interator, yielding
    batches of size `context_size`.

    Args:
        tokens_iterator: An iterator which returns a 1D tensors of tokens
        context_size: Each batch will have this many tokens
        begin_batch_token_id: If provided, this token will be at position 0 of each batch
        begin_sequence_token_id: If provided, this token will be the first token of each sequence
        sequence_separator_token_id: If provided, this token will be inserted between concatenated sequences
        max_batches: If not provided, the iterator will be run to completion.
    """
    batch: torch.Tensor | None = None
    for tokens in tokens_iterator:
        assert len(tokens.shape) == 1, f"tokens.shape should be 1D but was {tokens.shape}"
        offset = 0
        total_toks = tokens.shape[0]
        # is_start_of_sequence = True
        while total_toks - offset > 0:
            batch, offset = _add_tokens_to_batch(
                batch=batch,
                tokens=tokens,
                offset=offset,
                context_size=context_size,
                # is_start_of_sequence=is_start_of_sequence,
                begin_batch_token_id=begin_batch_token_id,
                begin_sequence_token_id=begin_sequence_token_id,
                sequence_separator_token_id=sequence_separator_token_id,
            )
            # is_start_of_sequence = False
            if batch.shape[0] == context_size:
                yield batch
                batch = None
