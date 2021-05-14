import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = math.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, d_model, dropout):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.w_q = Linear(q_dim, d_model)
        self.w_k = Linear(k_dim, d_model)
        self.w_v = Linear(v_dim, d_model)
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = Linear(d_model, d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, 2048, dropout)

    def forward(self, q, k, v):
        b_size = q.size(0)
        residual = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(b_size)
        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, v)
        output = self.dropout(self.proj(context))
        output = self.layer_norm(residual + output)
        output = self.pos_ffn(output)
        return output, attn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, d_model, n_heads, dropout):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = Linear(q_dim, d_model * n_heads)
        self.w_k = Linear(k_dim, d_model * n_heads)
        self.w_v = Linear(v_dim, d_model * n_heads)
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = Linear(n_heads * d_model, d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, 2048, dropout)

    def forward(self, q, k, v):
        b_size = q.size(0)
        residual = k
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_model).transpose(1, 2)

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / math.sqrt(b_size)
        attn = self.softmax(scores)
        context = torch.matmul(attn.transpose(2, 3), v_s)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_model)
        output = self.dropout(self.proj(context))
        """
        output = self.layer_norm(residual + output)
        output = self.pos_ffn(output)
        """
        return output, attn
