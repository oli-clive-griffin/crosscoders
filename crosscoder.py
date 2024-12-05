from dataclasses import dataclass
from einops import rearrange, einsum, reduce
import torch as t
from constants import DTYPE
from torch import nn

"""
Dimensions:
- N: batch size
- L: number of layers
- D: Subject model dimension
- I: Autoencoder hidden dimension, in line with anthropic's notation of f(x)_i meaning the i-th element of f(x)
"""

t.Tensor.d = lambda self: f"{self.shape}, dtype={self.dtype}, device={self.device}"  # type: ignore


@dataclass
class Losses:
    reconstruction_loss: t.Tensor
    sparsity_loss: t.Tensor


class CausalCrosscoder(nn.Module):
    """crosscoder that maps activations from a single layer to a subset of layers from that layer on (potentially including the input layer)"""

    def __init__(self, n_layers_out: int, d_model: int, hidden_dim: int):
        super().__init__()

        self.W_enc_ID = nn.Parameter(t.randn((hidden_dim, d_model)))  #, dtype=DTYPE))
        self.b_enc_I = nn.Parameter(t.zeros((hidden_dim,)))  #, dtype=DTYPE))

        self.W_dec_LDI = nn.Parameter(t.randn((n_layers_out, d_model, hidden_dim)))  #, dtype=DTYPE))
        self.b_dec_LD = nn.Parameter(t.zeros((n_layers_out, d_model)))  #, dtype=DTYPE))

    def encode(self, activation_ND: t.Tensor) -> t.Tensor:
        hidden_NI = einsum(activation_ND, self.W_enc_ID, "... d_model, hidden d_model -> ... hidden")
        hidden_NI += self.b_enc_I
        return t.relu(hidden_NI)

    def decode(self, hidden_NI: t.Tensor) -> t.Tensor:
        activation_NLD = einsum(hidden_NI, self.W_dec_LDI, "... hidden, layer d_model hidden -> ... layer d_model")
        activation_NLD += self.b_dec_LD
        return activation_NLD

    def forward(self, activation_ND: t.Tensor) -> t.Tensor:
        hidden_NI = self.encode(activation_ND)
        reconstructed_NLD = self.decode(hidden_NI)
        return reconstructed_NLD

    def forward_train(
        self,
        activation_ND: t.Tensor,
        future_activations_NLD: t.Tensor,
    ) -> tuple[t.Tensor, Losses]:
        hidden_NI = self.encode(activation_ND)
        reconstructed_NLD = self.decode(hidden_NI)
        reconstruction_loss_ = reconstruction_loss(reconstructed_NLD, future_activations_NLD)
        sparsity_loss_ = sparsity_loss(self.W_dec_LDI, hidden_NI)
        losses = Losses(reconstruction_loss=reconstruction_loss_, sparsity_loss=sparsity_loss_)
        return reconstructed_NLD, losses


class AcausalCrosscoder(nn.Module):
    """crosscoder that autoencodes activations of a subset of a model's layers"""

    def __init__(self, n_layers: int, d_model: int, hidden_dim: int, dec_init_norm: float):
        super().__init__()

        self.W_enc_LID = nn.Parameter(t.randn((n_layers, hidden_dim, d_model)))  #, dtype=DTYPE))
        self.b_enc_I = nn.Parameter(t.zeros((hidden_dim,)))  #, dtype=DTYPE))

        self.W_dec_LDI = nn.Parameter(t.randn((n_layers, d_model, hidden_dim)))  #, dtype=DTYPE))
        self.b_dec_LD = nn.Parameter(t.zeros((n_layers, d_model)))  #, dtype=DTYPE))

        with t.no_grad():
            W_dec_norm_LD1 = reduce(self.W_dec_LDI, "... hidden -> ... 1", lambda x, dims: t.norm(x, dim=dims))
            self.W_dec_LDI.div_(W_dec_norm_LD1)
            self.W_dec_LDI.mul_(dec_init_norm)

            # # Initialise W_enc to be the transpose of W_dec
            self.W_enc_LID.data = rearrange(
                self.W_dec_LDI.clone(),
                "layer d_model hidden  -> layer hidden d_model",
            )

    def encode(self, activation_NLD: t.Tensor) -> t.Tensor:
        """... d_model -> ... layer hidden"""
        # parallel matrix vector multiplication across layers
        hidden_NLI = einsum(activation_NLD, self.W_enc_LID, "... layer d_model, layer hidden d_model -> ... layer hidden")
        # sum layer encodings across layer dim
        hidden_NI = hidden_NLI.sum(1) + self.b_enc_I
        return t.relu(hidden_NI)

    def decode(self, hidden_NI: t.Tensor) -> t.Tensor:
        """... hidden -> ... layer d_model"""
        # parallel matrix vector multiplication across layers
        activation_NLD = einsum(hidden_NI, self.W_dec_LDI, "... hidden, ... layer d_model hidden -> ... layer d_model")
        activation_NLD += self.b_dec_LD
        return activation_NLD

    def forward(self, activation_NLD: t.Tensor) -> t.Tensor:
        """... layer d_model -> ... layer d_model"""
        hidden_NI = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NI)
        return reconstructed_NLD

    def forward_train(self, activation_NLD: t.Tensor) -> tuple[t.Tensor, Losses]:
        """... layer d_model -> ... layer d_model"""
        hidden_NI = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NI)

        losses = Losses(
            reconstruction_loss=reconstruction_loss(reconstructed_NLD, activation_NLD),
            sparsity_loss=sparsity_loss(self.W_dec_LDI, hidden_NI),
        )

        return reconstructed_NLD, losses


def reconstruction_loss(activation_NLD: t.Tensor, target_NLD: t.Tensor) -> t.Tensor:
    x_NL = (activation_NLD - target_NLD).norm(dim=-1).square()
    x_N = x_NL.sum(dim=-1)
    return x_N.mean()


def sparsity_loss(W_dec_LDI: t.Tensor, hidden_NI: t.Tensor) -> t.Tensor:
    """Computes the norm-weighted sparsity loss.
    for each decoder layer, compute the norm of the weights for each hidden dimension,
    then compute l1 norm of the activations, weighted by the decoder layer weights' norm.
    then, sum across layers, mean over batch.
    """
    assert (hidden_NI >= 0).all()
    W_dec_l2_norms_LI = W_dec_LDI.norm(dim=1, p=2)
    summed_norms_I = W_dec_l2_norms_LI.sum(dim=0)
    weighted_hidden_NI = hidden_NI * summed_norms_I
    # sum is acting as L1 norm here
    sparsity_loss = weighted_hidden_NI.sum(dim=1).mean()
    return sparsity_loss


def test():
    batch_size = 2
    n_layers = 4
    d_model = 16
    hidden_dim = 256

    crosscoder = AcausalCrosscoder(
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dec_init_norm=0.1,
    )
    activations_NLD = t.randn(batch_size, n_layers, d_model)  # , dtype=DTYPE)
    y_NLD = crosscoder.forward(activations_NLD)
    assert y_NLD.shape == activations_NLD.shape
    y_NLD, loss = crosscoder.forward_train(activations_NLD)
    assert y_NLD.shape == activations_NLD.shape
    assert loss.reconstruction_loss.shape == ()
    assert loss.sparsity_loss.shape == ()

    causal_crosscoder = CausalCrosscoder(
        n_layers_out=n_layers - 1,
        d_model=d_model,
        hidden_dim=hidden_dim,
    )
    activations_NLD = t.randn(batch_size, n_layers, d_model)  # , dtype=DTYPE)
    input_N1D, later_activations_NLD = activations_NLD.split([1, n_layers - 1], dim=1)
    input_ND = input_N1D.squeeze(1)
    assert input_ND.shape == (batch_size, d_model)
    assert later_activations_NLD.shape == (batch_size, n_layers - 1, d_model)

    y_NLD, loss = causal_crosscoder.forward_train(input_ND, later_activations_NLD)
    assert y_NLD.shape == later_activations_NLD.shape
    assert loss.reconstruction_loss.shape == ()
    assert loss.sparsity_loss.shape == ()


if __name__ == "__main__":
    t.set_default_dtype(DTYPE)
    test()
