import torch
import torch.nn as nn
import dgl

from .PwGaANLayer import MultiHeadPwGaANLayer


class CGNN(nn.Module):
    def __init__(self, in_dim, out_dim, use_pre_w, blk_size,
                 att, num_heads, merge, gate):
        super(CGNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_pre_w = use_pre_w
        self.blk_size = blk_size

        self.att = att
        self.num_heads = num_heads
        self.merge = merge
        self.gate = gate

        self.layers = nn.ModuleList([
            MultiHeadPwGaANLayer(self.in_dim, self.out_dim,
                                 self.num_heads, self.merge, self.att, self.gate, self.use_pre_w)
            for _ in range(self.blk_size)
        ])

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     """ Reinitialize learnable parameters. """
    #     gain = nn.init.calculate_gain('leaky_relu')
    #     nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)

    def forward(self, g: dgl.DGLGraph):
        return None


class GCN(CGNN):
    def __init__(self, in_dim, out_dim, use_pre_w, blk_size):
        super(GCN, self).__init__(in_dim=in_dim, out_dim=out_dim, use_pre_w=use_pre_w, blk_size=blk_size,
                                  att=False, num_heads=1, merge='cat', gate=False)


class GAT(CGNN):
    def __init__(self, in_dim, out_dim, use_pre_w, blk_size,
                 num_heads, merge):
        super(GAT, self).__init__(in_dim=in_dim, out_dim=out_dim, use_pre_w=use_pre_w, blk_size=blk_size,
                                  att=True, num_heads=num_heads, merge=merge, gate=False)


class GaAN(CGNN):
    def __init__(self, in_dim, out_dim, use_pre_w, blk_size,
                 num_heads, merge):
        super(GaAN, self).__init__(in_dim=in_dim, out_dim=out_dim, use_pre_w=use_pre_w, blk_size=blk_size,
                                   att=True, num_heads=num_heads, merge=merge, gate=True)


if __name__ == '__main__':
    """ Test """
    # TODO
    model = GaAN(in_dim=20, out_dim=5, use_pre_w=True, blk_size=2, num_heads=3, merge='cat')
    res = model(None)
