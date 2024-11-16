from dataclasses import dataclass
from einops import rearrange
import torch as t
from torch import nn
@dataclass
class Losses:
    reconstruction_loss: t.Tensor
    sparsity_loss: t.Tensor


class CausalCrosscoder(nn.Module):
    def __init__(self, n_layers_out: int, layer_dim: int, hidden_dim: int):
        super().__init__()

        self.W_enc_HD = nn.Parameter(t.randn(hidden_dim, layer_dim))
        self.b_enc_H = nn.Parameter(t.zeros(hidden_dim))

        self.W_dec_LDH = nn.Parameter(t.randn(n_layers_out, layer_dim, hidden_dim))
        self.b_dec_LD = nn.Parameter(t.zeros(n_layers_out, layer_dim))

    def encode(self, activation_ND: t.Tensor) -> t.Tensor:
        """Encodes activations from one layer.

        activation_ND: (batch_size, layer_dim)
        """
        hidden_NH = t.einsum("nd,hd->nh", activation_ND, self.W_enc_HD)
        hidden_NH += self.b_enc_H
        return t.relu(hidden_NH)

    def decode(self, hidden_NH: t.Tensor) -> t.Tensor:
        """
        hidden_NH: (batch_size, hidden_dim)
        """
        activation_NLD = t.einsum("nh,ldh->nld", hidden_NH, self.W_dec_LDH)
        activation_NLD += self.b_dec_LD
        return activation_NLD

    def forward(self, activation_ND: t.Tensor) -> t.Tensor:
        """
        activation_ND: (batch_size, layer_dim)
        """
        hidden_NH = self.encode(activation_ND)
        reconstructed_NLD = self.decode(hidden_NH)
        return reconstructed_NLD

    def forward_train(
        self,
        activation_ND: t.Tensor,
        future_activations_NLD: t.Tensor,
    ) -> tuple[t.Tensor, t.Tensor]:
        hidden_NH = self.encode(activation_ND)
        reconstructed_NLD = self.decode(hidden_NH)
        reconstruction_loss_ = reconstruction_loss(reconstructed_NLD, future_activations_NLD)
        sparsity_loss_ = sparsity_loss(self.W_dec_LDH, hidden_NH)
        loss = reconstruction_loss_ + sparsity_loss_
        return reconstructed_NLD, loss


class AcausalCrosscoder(nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, hidden_dim: int, dec_init_norm: float):
        super().__init__()

        self.W_enc_LHD = nn.Parameter(t.randn(n_layers, hidden_dim, layer_dim))
        self.b_enc_H = nn.Parameter(t.zeros(hidden_dim))

        self.W_dec_LDH = nn.Parameter(t.randn(n_layers, layer_dim, hidden_dim))
        self.b_dec_LD = nn.Parameter(t.zeros(n_layers, layer_dim))

        self.lambda_ = 1

        with t.no_grad():
            self.W_dec_LDH.div_(self.W_dec_LDH.norm(dim=-1, keepdim=True))
            self.W_dec_LDH.mul_(dec_init_norm)

            # # Initialise W_enc to be the transpose of W_dec
            self.W_enc_LHD.data = rearrange(
                self.W_dec_LDH.clone(),
                "n_layers d_model d_hidden  -> n_layers d_hidden d_model",
            )



    def encode(self, activation_NLD: t.Tensor) -> t.Tensor:
        """
        activation_NLD: (batch_size, n_layers, layer_dim)
        """
        hidden_NLH = t.einsum("nld,lhd->nlh", activation_NLD, self.W_enc_LHD)
        hidden_NH = t.einsum("nlh->nh", hidden_NLH) + self.b_enc_H
        return t.relu(hidden_NH)

    def decode(self, hidden_NH: t.Tensor) -> t.Tensor:
        """
        hidden_NH: (batch_size, hidden_dim)
        """
        activation_NLD = t.einsum("nh,ldh->nld", hidden_NH, self.W_dec_LDH)
        activation_NLD += self.b_dec_LD
        return activation_NLD

    def forward(self, activation_NLD: t.Tensor) -> t.Tensor:
        """
        activation_NLD: (batch_size, n_layers, layer_dim)
        """
        hidden_NH = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NH)
        return reconstructed_NLD

    def forward_train(self, activation_NLD: t.Tensor) -> tuple[t.Tensor, Losses]:
        hidden_NH = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NH)

        losses = Losses(
            reconstruction_loss=reconstruction_loss(reconstructed_NLD, activation_NLD),
            sparsity_loss=sparsity_loss(self.W_dec_LDH, hidden_NH),
        )

        return reconstructed_NLD, losses




def reconstruction_loss(activation_NLD: t.Tensor, target_NLD: t.Tensor) -> t.Tensor:
    x_NL = (activation_NLD - target_NLD).norm(dim=-1).square()
    x_N = x_NL.sum(dim=-1)
    return x_N.mean()


def sparsity_loss(W_dec_LDH: t.Tensor, hidden_NH: t.Tensor) -> t.Tensor:
    W_dec_norm_LH = W_dec_LDH.norm(dim=1)
    W_dec_norm_H = t.einsum("lh->h", W_dec_norm_LH)
    return t.einsum("nh,h->n", hidden_NH, W_dec_norm_H).mean()


if __name__ == "__main__":
    batch_size = 1
    n_layers = 4
    layer_dim = 16
    hidden_dim = 256

    crosscoder = AcausalCrosscoder(
        n_layers=n_layers,
        layer_dim=layer_dim,
        hidden_dim=hidden_dim,
    )
    activations_NLD = t.randn(batch_size, n_layers, layer_dim)
    y_NLD = crosscoder.forward(activations_NLD)
    assert y_NLD.shape == activations_NLD.shape
    y_NLD, loss = crosscoder.forward_train(activations_NLD)
    assert y_NLD.shape == activations_NLD.shape
    assert loss.shape == ()

    causal_crosscoder = CausalCrosscoder(
        n_layers_out=n_layers - 1,
        layer_dim=layer_dim,
        hidden_dim=hidden_dim,
    )
    activations_NLD = t.randn(batch_size, n_layers, layer_dim)
    input_N1D, later_activations_NLD = activations_NLD.split([1, n_layers - 1], dim=1)
    input_ND = input_N1D.squeeze(1)
    assert input_ND.shape == (batch_size, layer_dim)
    assert later_activations_NLD.shape == (batch_size, n_layers - 1, layer_dim)

    y_NLD, loss = causal_crosscoder.forward_train(input_ND, later_activations_NLD)
    assert y_NLD.shape == later_activations_NLD.shape
    assert loss.shape == ()
