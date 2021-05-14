import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import GraphConvolution
from HighChebNet import HighChebNet
from CrossAttention import LayerNormalization, PositionalEncoding


class HDGCN(nn.Module):
    def __init__(self,
                 nnodes,
                 nfeat,
                 nhid,
                 nclass,
                 max_seq_len,
                 device,
                 batch_size,
                 vocab):
        super(HDGCN, self).__init__()

        self.reconstructed_image_count = 0
        self.device = device
        self.nnodes = nnodes
        self.max_seq_len = max_seq_len
        self.nfeat = nfeat
        self.vocab = vocab

        A = 10
        self.pe = PositionalEncoding(nfeat, 0.1)
        self.layernorm = LayerNormalization(64)
        self.PrimeChebNet1 = GraphConvolution(nfeat, nhid, nnodes=nnodes, device=device)
        self.PrimeChebNet2 = GraphConvolution(nhid, 64, nnodes=nnodes, device=device)
        self.HiChebNet1 = HighChebNet(in_units=nnodes, out_units=A, in_dim=64, vote_dim=64, batch_size=batch_size,
                                      device=device)
        self.PrimeChebNet3 = GraphConvolution(64, 64, nnodes=nnodes, device=device)
        self.PrimeChebNet4 = GraphConvolution(64, 64, nnodes=nnodes, device=device)
        self.HiChebNet2 = HighChebNet(in_units=nnodes, out_units=A, in_dim=64, vote_dim=64, batch_size=batch_size,
                                      device=device)
        self.PrimeChebNet5 = GraphConvolution(64, 64, nnodes=nnodes, device=device)
        self.PrimeChebNet6 = GraphConvolution(64, 64, nnodes=nnodes, device=device)
        self.mlp1 = nn.Linear(nfeat + 5 * 64, 64)
        self.mlp2 = nn.Linear(nfeat + 5 * 64, 64)
        self.logit = nn.Linear(64, nclass)

    def forward(self, x, adj, sentence, batch, vis=False):
        x = self.pe(x)
        out1 = F.relu(self.PrimeChebNet1(x, adj))
        out1 = F.dropout(out1, 0.3, training=self.training)
        out1 = F.relu(self.PrimeChebNet2(out1, adj))
        out1 = F.dropout(out1, 0.3, training=self.training)
        hicheb_out1, _, _, _ = self.HiChebNet1(out1)
        out2 = F.relu(self.PrimeChebNet3(hicheb_out1, adj))
        out2 = F.dropout(out2, 0.3, training=self.training)
        out2 = F.relu(self.PrimeChebNet4(out2, adj))
        out2 = F.dropout(out2, 0.3, training=self.training)
        hicheb_out2, _, _, _ = self.HiChebNet2(out2)
        out3 = F.relu(self.gcn5(hicheb_out2, adj))
        out3 = F.dropout(out3, 0.3, training=self.training)
        out3 = self.gcn6(out3, adj)
        out3 = F.dropout(out3, 0.3, training=self.training)
        output = torch.cat((x, out1, hicheb_out1, out2, hicheb_out2, out3), dim=-1)
        output = F.dropout(torch.sigmoid(self.mlp1(output)) * F.relu(self.mlp2(output)), 0.5, training=self.training)
        output = self.layernorm(
            output.mean(dim=1, keepdim=False) + F.avg_pool2d(output, (output.shape[1], 1)).squeeze(1))
        output = self.logit(output)
        return F.log_softmax(output, dim=-1)
