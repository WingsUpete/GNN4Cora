import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .CGaANLayer import MultiHeadCGaANLayer

import Config


class CGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_view, blk_size,
                 att, num_heads, merge, gate):
        super(CGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.num_view = num_view
        self.blk_size = blk_size

        self.att = att
        self.num_heads = num_heads
        self.merge = merge  # the merge mode of the last block, former blocks are all "mean"
        self.gate = gate

        self.proj_fc = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.embed_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.layers = nn.ModuleList([nn.ModuleList(
            ([MultiHeadCGaANLayer(self.hidden_dim, self.hidden_dim,
                                  self.att, self.num_heads, 'mean', self.gate)
             for _ in range(self.blk_size - 1)] if self.blk_size > 1 else []) +
            [MultiHeadCGaANLayer(self.hidden_dim, self.hidden_dim,
                                 self.att, self.num_heads, self.merge, self.gate)]
        ) for _ in range(self.num_view)])

        if self.merge == 'mean':
            self.tran_dim = int(self.hidden_dim * (1 + self.num_view))
        elif self.merge == 'cat':
            self.tran_dim = int(self.hidden_dim * (1 + (self.num_heads * self.num_view)))
        else:
            self.tran_dim = int(self.hidden_dim * (1 + (self.num_heads * self.num_view)))     # Default: cat
        self.tran_fc = nn.Linear(self.tran_dim, self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.embed_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.tran_fc.weight, gain=gain)

    def forward(self, gs: list, ft: torch.Tensor):
        # Reduce dimension
        proj_feat = self.proj_fc(ft)
        proj_feats = [proj_feat for _ in range(self.num_view)]

        cur_feats = proj_feats
        for i in range(self.blk_size):
            # Assign "v" and "proj_z"
            embed_feats = [self.embed_fc(cur_feat) for cur_feat in cur_feats]
            for j in range(self.num_view):
                gs[j].ndata['v'] = cur_feats[j]
                gs[j].ndata['proj_z'] = embed_feats[j]
            # Convolution to get output as feature for the next block
            cur_feats = [self.layers[j][i](gs[j]) for j in range(self.num_view)]
            cur_feats = [F.leaky_relu(cur_feats[j]) for j in range(self.num_view)]

        h = torch.cat([proj_feat] + cur_feats, dim=-1)
        del embed_feats
        del cur_feats

        out = self.tran_fc(h)
        return out


class GCN(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_view, blk_size):
        super(GCN, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                  num_view=num_view, blk_size=blk_size,
                                  att=False, num_heads=1, merge='cat', gate=False)


class GAT(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_view, blk_size,
                 num_heads, merge):
        super(GAT, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                  num_view=num_view, blk_size=blk_size,
                                  att=True, num_heads=num_heads, merge=merge, gate=False)


class GaAN(CGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_view, blk_size,
                 num_heads, merge):
        super(GaAN, self).__init__(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                   num_view=num_view, blk_size=blk_size,
                                   att=True, num_heads=num_heads, merge=merge, gate=True)


if __name__ == '__main__':
    """ 
    Test: Remember to remove '.' of 'from .CGaANLayer import MultiHeadCGaANLayer' when testing and add back afterwards 
    """
    (citing_graph, cited_graph, both_graph), _ = dgl.load_graphs('../data/cora/cora.dgl')
    feats = citing_graph.ndata['feat']
    # model = GCN(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
    #              num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT)
    # model = GAT(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
    #              num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
    #              num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    model = GaAN(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES,
                 num_view=Config.NUM_VIEW_DEFAULT, blk_size=Config.BLK_SIZE_DEFAULT,
                 num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT)
    res = model([both_graph], feats)
    print(res.shape)
