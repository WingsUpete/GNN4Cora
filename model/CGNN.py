import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .CGaANLayer import MultiHeadCGaANLayer

import Config


class CGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, blk_size,
                 att, num_heads, merge, gate):
        super(CGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.blk_size = blk_size

        self.att = att
        self.num_heads = num_heads
        self.merge = merge  # the merge mode of the last block, former blocks are all "mean"
        self.gate = gate

        self.proj_fc = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.embed_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.layers = nn.ModuleList(
            ([MultiHeadCGaANLayer(self.hidden_dim, self.hidden_dim,
                                 self.att, self.num_heads, 'mean', self.gate)
             for _ in range(self.blk_size - 1)] if self.blk_size > 1 else []) +
            [MultiHeadCGaANLayer(self.hidden_dim, self.hidden_dim,
                                 self.att, self.num_heads, self.merge, self.gate)]
        )

        if self.merge == 'mean':
            self.tran_dim = int(self.hidden_dim * 2)
        elif self.merge == 'cat':
            self.tran_dim = int(self.hidden_dim * (self.num_heads + 1))
        else:
            self.tran_dim = int(self.hidden_dim * (self.num_heads + 1))     # Default: cat
        self.tran_fc = nn.Linear(self.tran_dim, self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.embed_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.tran_fc.weight, gain=gain)

    def forward(self, g: dgl.DGLGraph, ft: torch.Tensor):
        # Reduce dimension
        proj_feat = self.proj_fc(ft)

        cur_feat = proj_feat
        for i in range(self.blk_size):
            # Assign "v" and "proj_z"
            g.ndata['v'] = cur_feat
            embed_feat = self.embed_fc(cur_feat)
            g.ndata['proj_z'] = embed_feat
            # Convolution to get output as feature for the next block
            cur_feat = self.layers[i](g)
            cur_feat = F.leaky_relu(cur_feat)

        h = torch.cat([proj_feat, cur_feat], dim=-1)
        del embed_feat
        del cur_feat

        out = self.tran_fc(h)
        return out


class GCN(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, blk_size):
        super(GCN, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, blk_size=blk_size,
                                  att=False, num_heads=1, merge='cat', gate=False)


class GAT(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, blk_size,
                 num_heads, merge):
        super(GAT, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, blk_size=blk_size,
                                  att=True, num_heads=num_heads, merge=merge, gate=False)


class GaAN(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, blk_size,
                 num_heads, merge):
        super(GaAN, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, blk_size=blk_size,
                                   att=True, num_heads=num_heads, merge=merge, gate=True)


if __name__ == '__main__':
    """ 
    Test: Remember to remove '.' of 'from .CGaANLayer import MultiHeadCGaANLayer' when testing and add back afterwards 
    """
    (graph,), _ = dgl.load_graphs('../data/cora/cora.dgl')
    feats = graph.ndata['feat']
    # model = GCN(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
    #              blk_size=Config.BLK_SIZE_DEFAULT)
    # model = GAT(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
    #              blk_size=Config.BLK_SIZE_DEFAULT, num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    model = GaAN(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
                 blk_size=Config.BLK_SIZE_DEFAULT, num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    res = model(graph, feats)
    print(res.shape)
