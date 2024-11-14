import torch as t
from torch import nn


class CrossCoder(nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, hidden_dim: int):
        super(CrossCoder, self).__init__()

        self.enc_weights_LHF = nn.Parameter(t.randn(n_layers, hidden_dim, layer_dim))
        self.enc_bias_H = nn.Parameter(t.zeros(hidden_dim))

        self.dec_weights_LFH = nn.Parameter(t.randn(n_layers, layer_dim, hidden_dim))
        self.dec_bias_LF = nn.Parameter(t.zeros(n_layers, layer_dim))

    def encode(self, activation_NLF: t.Tensor) -> t.Tensor:
        """
        activation_NLF: (batch_size, n_layers, layer_dim)
        """
        hidden_NLH = t.einsum("nlf,lhf->nlh", activation_NLF, self.enc_weights_LHF)
        hidden_NH = t.einsum("nlh->nh", hidden_NLH) + self.enc_bias_H
        return t.relu(hidden_NH)

    def decode(self, hidden_NH: t.Tensor) -> t.Tensor:
        """
        hidden_NH: (batch_size, n_layers)
        """
        activation_NLF = t.einsum("nh,lfh->nlf", hidden_NH, self.dec_weights_LFH)
        activation_NLF += self.dec_bias_LF
        return activation_NLF

    def forward(self, activation_NLF):
        hidden_NH = self.encode(activation_NLF)
        reconstructed_NLF = self.decode(hidden_NH)
        return reconstructed_NLF

    def sparsity_loss(self, hidden_NH: t.Tensor) -> t.Tensor:
        dec_weights_norm_LH = self.dec_weights_LFH.norm(dim=1)
        dec_weights_norm_H = t.einsum("lh->h", dec_weights_norm_LH)
        return t.einsum("nh,h->n", hidden_NH, dec_weights_norm_H).mean()

    def reconstruction_loss(
        self, activation_NLF: t.Tensor, target_NLF: t.Tensor
    ) -> t.Tensor:
        x_NL = (activation_NLF - target_NLF).norm(dim=-1).square()
        x_N = t.einsum("nl->n", x_NL)
        return x_N.mean()

    def forward_train(self, activation_NLF: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        hidden_NH = self.encode(activation_NLF)
        reconstructed_NLF = self.decode(hidden_NH)
        reconstruction_loss = self.reconstruction_loss(reconstructed_NLF, activation_NLF)
        sparsity_loss = self.sparsity_loss(hidden_NH)
        loss = reconstruction_loss + sparsity_loss
        return reconstructed_NLF, loss


if __name__ == "__main__":
    batch_size = 1
    n_layers = 4
    layer_dim = 16
    hidden_dim = 256
    model = CrossCoder(n_layers=n_layers, layer_dim=layer_dim, hidden_dim=hidden_dim)
    x_NLF = t.randn(batch_size, n_layers, layer_dim)
    y_NLF = model.forward(x_NLF)
    assert y_NLF.shape == x_NLF.shape

    y_NLF, loss = model.forward_train(x_NLF)
    assert y_NLF.shape == x_NLF.shape
    assert loss.shape == ()
