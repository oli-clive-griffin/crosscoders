from typing import Iterator

import einops
import numpy as np
import torch
from datasets import Dataset, load_dataset  # type: ignore

from crosscoder import AcausalCrosscoder
from llm import LLMWithCache


def train():
    llm = LLMWithCache()
    crosscoder = AcausalCrosscoder(n_layers=23, layer_dim=2048, hidden_dim=2**10)  # * 8)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=1e-3)
    dataset = load_dataset("PleIAs/common_corpus", streaming=True)
    dataloader = iter_dataset(dataset)
    for batch in dataloader:
        optimizer.zero_grad()
        hidden_states = llm.get_residual_stream(batch)
        hidden_states_NSLD = torch.from_numpy(np.stack(hidden_states))
        hidden_states_NLD = einops.rearrange(hidden_states_NSLD, "n s l d -> (n s) l d")
        hidden_states_NLD, loss = crosscoder.forward_train(hidden_states_NLD)
        print(f"loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()


def iter_dataset(
    ds: Dataset, batch_size: int = 4, sequence_length: int = 1024
) -> Iterator[list[str]]:
    for example in ds["train"]:
        for i in range(0, len(example["text"]), batch_size * sequence_length):
            text = example["text"][i : i + batch_size * sequence_length]
            yield [text[j : j + sequence_length] for j in range(0, len(text), sequence_length)]


if __name__ == "__main__":
    train()
