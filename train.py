from typing import Iterator

import einops
import einops.experimental
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from datasets import Dataset, load_dataset  # type: ignore
from transformers import PreTrainedTokenizer, AutoTokenizer  # type: ignore

from crosscoder import AcausalCrosscoder
from llm import LLMWithCache

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def train():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    llm = LLMWithCache(model_name).to(device)
    crosscoder = AcausalCrosscoder(n_layers=23, layer_dim=2048, hidden_dim=32_768).to(device)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=1e-3)

    for batch_NS in get_dataloader(model_name, batch_size=16, sequence_length=64):
        batch_NS = batch_NS.to(device)
        optimizer.zero_grad()
        hidden_states_NLD = llm.get_residual_stream_NLD(batch_NS)
        hidden_states_NLD = torch.tensor(hidden_states_NLD, dtype=torch.bfloat16).to(device)
        _, loss = crosscoder.forward_train(hidden_states_NLD)
        print(f"loss: {loss.item():.4f}")
        clip_grad_norm_(crosscoder.parameters(), 1.0)
        loss.backward()
        optimizer.step()


# TODO use buffer to shuffle and draw from buffer
def get_dataloader(model_name: str, batch_size: int, sequence_length: int) -> Iterator[torch.Tensor]:
    dataset = load_dataset("PleIAs/common_corpus", streaming=True, cache_dir="tmp/cache")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>")
    batch_length = batch_size * sequence_length

    for example in dataset["train"]:
        text_tokens = torch.tensor(tokenizer(example["text"])["input_ids"])

        n_tokens_rounded = round_down(len(text_tokens), batch_length)
        text_tokens = text_tokens[:n_tokens_rounded]

        for batch_Ns in torch.split(text_tokens, batch_length):
            batch_NS = einops.rearrange(batch_Ns, "(n s) -> n s", n=batch_size)
            yield batch_NS


def round_down(n: int, multiple: int) -> int:
    return (n // multiple) * multiple


if __name__ == "__main__":
    train()
