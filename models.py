import torch
import torch.nn as nn
from modules import Linear, Encoder


class Classifier(nn.Module):
    def __init__(self,
                 emb_size, vocab_size, label_size,
                 n_l, n_h, d_m, d_h, d_k, d_v, dropout):

        super(Classifier, self).__init__()

        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.linear = Linear(d_m, label_size)
        self.encoder = Encoder(n_h, d_m, d_h, d_k, d_v, dropout)
        self.softmax = nn.Softmax()

    def forward(self, _input):
        _len = _input.size(1)

        output, attn = self.encoder(_input)
        output = self.linear(output).view(-1, self.label_size)
        output = self.softmax(output).view(-1, _len, self.label_size)

        return torch.mean(output, dim=-1)
