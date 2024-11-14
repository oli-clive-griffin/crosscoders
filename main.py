import torch as t
from torch import nn


class CrossCoder(nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, hidden_dim: int):
        super(CrossCoder, self).__init__()

        self.enc_weights_LHF = nn.Parameter(t.zeros(n_layers, hidden_dim, layer_dim))
        t.nn.init.kaiming_uniform_(self.enc_weights_LHF)
        self.enc_bias_H = nn.Parameter(t.zeros(hidden_dim))

        self.dec_weights_LFH = nn.Parameter(t.zeros(n_layers, layer_dim, hidden_dim))
        t.nn.init.kaiming_uniform_(self.dec_weights_LFH)
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

    def forward(self, activation_NLF: t.Tensor) -> t.Tensor:
        return self.decode(self.encode(activation_NLF))


if __name__ == "__main__":
    batch_size = 1
    n_layers = 4
    layer_dim = 16
    hidden_dim = 256
    model = CrossCoder(n_layers=n_layers, layer_dim=layer_dim, hidden_dim=hidden_dim)
    x = t.randn(batch_size, n_layers, layer_dim)
    y = model.forward(x)
    assert y.shape == (batch_size, n_layers, layer_dim)
