import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features

        init.xavier_normal(self.linear.weight)

    def forward(self, _input):
        assert _input.dim() == 3, 'Requires a 3D tensor as input'

        _len = _input.size(1)

        output = self.linear(_input.view(-1, self.in_features))
        output = output.view(-1, _len, self.out_features)

        return output


class LayerNormalization(nn.Module):
    def __init__(self, d_h, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_h))
        self.b = nn.Parameter(torch.zeros(d_h))

    def forward(self, a):
        if a.size(1) == 1:
            return a

        mu = torch.mean(a, dim=-1)
        sigma = torch.std(a, dim=-1)

        out = (a - mu.expand_as(a)) / (sigma.expand_as(a) + self.eps)
        out = out * self.g.expand_as(out) + self.b.expand_as(out)

        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()

        self.temper = pow(d_k, .5)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        len_q = q.size(1)
        len_k = k.size(1)

        if mask is not None:
            # TODO: add masking feature.
            pass

        attn = self.softmax(attn.view(-1, len_k))
        attn = self.dropout(attn.view(-1, len_q, len_k))
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_h, d_m, d_k, d_v, dropout):
        super(MultiHeadAttention, self).__init__()

        self.n_h = n_h
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Parameter(torch.FloatTensor(n_h, d_m, d_k))
        self.W_k = nn.Parameter(torch.FloatTensor(n_h, d_m, d_k))
        self.W_v = nn.Parameter(torch.FloatTensor(n_h, d_m, d_v))

        self.attn = ScaledDotProductAttention(d_m, dropout)
        self.norm = LayerNormalization(d_m)
        self.linear = Linear(n_h * d_v, d_m)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.W_q)
        init.xavier_normal(self.W_k)
        init.xavier_normal(self.W_v)

    def forward(self, q, k, v, mask=None):
        b_size, len_q, d_m = q.size()
        b_size, len_k, d_m = k.size()

        n_h, d_k, d_v = self.n_h, self.d_k, self.d_v

        residual = q

        qs = q.repeat(n_h, 1, 1).view(n_h, -1, d_m)
        ks = k.repeat(n_h, 1, 1).view(n_h, -1, d_m)
        vs = k.repeat(n_h, 1, 1).view(n_h, -1, d_m)

        qs = torch.bmm(qs, self.W_q).view(-1, len_q, d_k)
        ks = torch.bmm(ks, self.W_k).view(-1, len_k, d_k)
        vs = torch.bmm(vs, self.W_v).view(-1, len_k, d_v)

        output, attn = self.attn(qs, ks, vs, mask)

        output = torch.cat(torch.split(output, b_size, dim=0), dim=-1)
        output = self.linear(output)
        output = self.dropout(output)

        return self.norm(output + residual), attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_m, d_h, dropout):
        super(PositionWiseFeedForward, self).__init__()

        self.W_1 = nn.Conv1d(d_m, d_h, 1)
        self.W_2 = nn.Conv1d(d_h, d_m, 1)
        self.norm = LayerNormalization(d_m)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        output = self.relu(self.W_1(x.transpose(1, 2)))
        output = self.W_2(output).transpose(2, 1)
        output = self.dropout(output)

        return self.norm(output + residual)


class Encoder(nn.Module):
    def __init__(self, n_h, d_m, d_h, d_k, d_v, dropout):
        super(Encoder, self).__init__()

        self.attn = MultiHeadAttention(n_h, d_m, d_k, d_v, dropout)
        self.pffn = PositionWiseFeedForward(d_m, d_h, dropout)

    def forward(self, _input, mask=None):
        output, attn = self.attn(_input, _input, _input, mask)
        output = self.pffn(output)

        return output, attn
