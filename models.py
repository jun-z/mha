import torch
import torch.nn as nn
from modules import Encoder, PositionalEncoding


class Classifier(nn.Module):
    def __init__(self,
                 layers, length,
                 vocab_size, label_size,
                 n_h, d_m, d_h, d_k, d_v, dropout):

        super(Classifier, self).__init__()

        self.vocab_size = vocab_size
        self.label_size = label_size

        self.enc = PositionalEncoding(d_m, length)
        self.emb = nn.Embedding(vocab_size, d_m)
        self.linear = nn.Linear(d_m, label_size, bias=False)

        self.encoders = []
        for _ in range(layers):
            self.encoders.append(Encoder(n_h, d_m, d_h, d_k, d_v, dropout))

    def forward(self, _input):
        output = self.enc(self.emb(_input))

        for encoder in self.encoders:
            output, attn = encoder(output)

        output = torch.squeeze(torch.mean(output, dim=1))
        output = self.linear(output)

        return output
