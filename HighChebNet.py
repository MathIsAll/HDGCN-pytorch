import torch
import torch.nn as nn
from CrossAttention import CrossAttention, MultiHeadCrossAttention, LayerNormalization, PositionalEncoding


class HighChebNet(nn.Module):

    def __init__(self, in_units, out_units, in_dim, vote_dim, batch_size, device):
        super(HighChebNet, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.in_units = in_units
        self.out_units = out_units
        self.in_dim = in_dim
        self.vote_dim = vote_dim
        self.layerNorm = LayerNormalization(vote_dim)
        self.layerNorm_input = LayerNormalization(in_dim)
        self.pos_emb = PositionalEncoding(in_dim, 0.1)
        self.proj = nn.Linear(in_dim, in_dim)
        self.transform_weights = nn.Parameter(torch.randn(1, in_units, out_units, vote_dim, in_dim))
        self.forward_attention = MultiHeadCrossAttention(q_dim=in_dim, k_dim=vote_dim, v_dim=in_dim,
                                                                d_model=vote_dim, n_heads=1, dropout=0.5)
        self.feedback_attention = MultiHeadCrossAttention(q_dim=in_dim, k_dim=vote_dim, v_dim=vote_dim,
                                                                 d_model=in_dim, n_heads=1, dropout=0.5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        value = x
        x = x.unsqueeze(-2)
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, self.out_units, 1, 1)
        votes = self.transform_weights.matmul(x)
        norm_value = self.layerNorm(votes.mean(dim=1, keepdim=False).squeeze(-1))
        aggregated_value, forward_attn_score = self.forward_attention(value, norm_value, value)
        diffused_value, backward_attn_score = self.feedback_cross_attentions(aggregated_value, value, aggregated_value)
        return diffused_value, aggregated_value, forward_attn_score, backward_attn_score
