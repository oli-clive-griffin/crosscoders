import torch as t
from torch import nn

class CrossCoder(nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, hidden_dim: int):
        super(CrossCoder, self).__init__()

        self.enc_weights_LHD = nn.Parameter(t.randn(n_layers, hidden_dim, layer_dim))
        self.enc_bias_H = nn.Parameter(t.zeros(hidden_dim))

        self.dec_weights_LDH = nn.Parameter(t.randn(n_layers, layer_dim, hidden_dim))
        self.dec_bias_LD = nn.Parameter(t.zeros(n_layers, layer_dim))

    def encode(self, activation_NLD: t.Tensor) -> t.Tensor:
        """
        activation_NLD: (batch_size, n_layers, layer_dim)
        """
        hidden_NLH = t.einsum("nld,lhd->nlh", activation_NLD, self.enc_weights_LHD)
        hidden_NH = t.einsum("nlh->nh", hidden_NLH) + self.enc_bias_H
        return t.relu(hidden_NH)

    def decode(self, hidden_NH: t.Tensor) -> t.Tensor:
        """
        hidden_NH: (batch_size, hidden_dim)
        """
        activation_NLD = t.einsum("nh,ldh->nld", hidden_NH, self.dec_weights_LDH)
        activation_NLD += self.dec_bias_LD
        return activation_NLD

    def forward(self, activation_NLD: t.Tensor) -> t.Tensor:
        """
        activation_NLD: (batch_size, n_layers, layer_dim)
        """
        hidden_NH = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NH)
        return reconstructed_NLD

    def sparsity_loss(self, hidden_NH: t.Tensor) -> t.Tensor:
        dec_weights_norm_LH = self.dec_weights_LDH.norm(dim=1)
        dec_weights_norm_H = t.einsum("lh->h", dec_weights_norm_LH)
        return t.einsum("nh,h->n", hidden_NH, dec_weights_norm_H).mean()

    def reconstruction_loss(
        self, activation_NLD: t.Tensor, target_NLD: t.Tensor
    ) -> t.Tensor:
        x_NL = (activation_NLD - target_NLD).norm(dim=-1).square()
        x_N = t.einsum("nl->n", x_NL)
        return x_N.mean()

    def forward_train(self, activation_NLD: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        hidden_NH = self.encode(activation_NLD)
        reconstructed_NLD = self.decode(hidden_NH)
        reconstruction_loss = self.reconstruction_loss(
            reconstructed_NLD, activation_NLD
        )
        sparsity_loss = self.sparsity_loss(hidden_NH)
        loss = reconstruction_loss + sparsity_loss
        return reconstructed_NLD, loss


if __name__ == "__main__":
    batch_size = 1
    n_layers = 4
    layer_dim = 16
    hidden_dim = 256
    model = CrossCoder(n_layers=n_layers, layer_dim=layer_dim, hidden_dim=hidden_dim)
    x_NLD = t.randn(batch_size, n_layers, layer_dim)
    y_NLD = model.forward(x_NLD)
    assert y_NLD.shape == x_NLD.shape

    y_NLD, loss = model.forward_train(x_NLD)
    assert y_NLD.shape == x_NLD.shape
    assert loss.shape == ()
