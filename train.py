from dataclasses import dataclass
from functools import partial
import einops
from transformer_lens import HookedTransformer  # type: ignore
import torch
from torch.nn.utils import clip_grad_norm_

from constants import DTYPE
from crosscoder import AcausalCrosscoder
from data import BufferedActivationLoader


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# MODEL_NAME = "google/gemma-2-2b"
MODEL_NAME = "gpt2"
CACHE_DIR = "tmp/cache"


@dataclass
class TrainConfig:
    n_layers_to_encode: int
    batch_size: int
    sequence_length: int
    buffer_n_sequences: int
    crosscoder_hidden_dim: int
    dec_init_norm = 0.1
    lr = 5e-5
    lambda_max = 5.0
    lambda_n_steps = 1000


def train(cfg: TrainConfig):
    llm: HookedTransformer = HookedTransformer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, dtype=DTYPE).to(DEVICE)
    print([name for name, _ in llm.named_parameters()])

    crosscoder = AcausalCrosscoder(
        n_layers=cfg.n_layers_to_encode,
        d_model=llm.cfg.d_model,
        hidden_dim=cfg.crosscoder_hidden_dim,
        dec_init_norm=cfg.dec_init_norm,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=cfg.lr)
    dataloader = BufferedActivationLoader(MODEL_NAME, cfg.batch_size, cfg.sequence_length, cfg.buffer_n_sequences)
    lambda_step = partial(lambda_scheduler, cfg.lambda_max, cfg.lambda_n_steps)

    for step, batch_Ns_LD in enumerate(dataloader):
        batch_Ns_LD = batch_Ns_LD.to(DEVICE)
        optimizer.zero_grad()

        _, losses = crosscoder.forward_train(batch_Ns_LD)

        lambda_ = lambda_step(step)
        loss = losses.reconstruction_loss + lambda_ * losses.sparsity_loss

        print(
            f"Step {step:05d}, loss: {loss.item():.4f}, lambda: {lambda_:.4f}, reconstruction_loss: {losses.reconstruction_loss.item():.4f}, sparsity_loss: {losses.sparsity_loss.item():.4f}"
        )

        clip_grad_norm_(crosscoder.parameters(), 1.0)
        loss.backward()
        optimizer.step()


def lambda_scheduler(lambda_max: float, n_steps: int, step_: int):
    if step_ < n_steps:
        return lambda_max * step_ / n_steps
    else:
        return lambda_max


if __name__ == "__main__":
    cfg = TrainConfig(
        n_layers_to_encode=4,
        batch_size=128,
        sequence_length=256,
        buffer_n_sequences=500,
        crosscoder_hidden_dim=32_768,
    )
    train(cfg)
