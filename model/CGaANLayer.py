import sys

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import Config


class CGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, att=True, gate=True):
        super(CGaANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.att = att
        if self.att:
            # Shared Weight W_a for AttentionNet
            self.Wa = nn.Linear(self.in_dim, self.out_dim, bias=False)
            # AttentionNet outer linear layer
            # split fc to avoid cat
            self.att_out_fc_l = nn.Linear(self.out_dim, 1, bias=False)
            self.att_out_fc_r = nn.Linear(self.out_dim, 1, bias=False)

        # Head gate layer
        self.gate = gate
        if self.gate:
            # split fc to avoid cat
            self.gate_fc_l = nn.Linear(self.in_dim, 1, bias=False)
            self.gate_fc_m = nn.Linear(self.out_dim, 1, bias=False)
            self.gate_fc_r = nn.Linear(self.in_dim, 1, bias=False)
            self.Wgm = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        if self.att:
            nn.init.xavier_normal_(self.Wa.weight, gain=gain)
            nn.init.xavier_normal_(self.att_out_fc_l.weight, gain=gain)
            nn.init.xavier_normal_(self.att_out_fc_r.weight, gain=gain)
        if self.gate:
            gain = nn.init.calculate_gain('sigmoid')
            nn.init.xavier_normal_(self.gate_fc_l.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc_m.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc_r.weight, gain=gain)
            nn.init.xavier_normal_(self.Wgm.weight, gain=gain)

    def edge_attention(self, edges):
        a = self.att_out_fc_l(edges.src['z']) + self.att_out_fc_r(edges.dst['z'])
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """ Specify messages to be propagated along edges """
        # The messages will be sent to the mailbox
        # mailbox['proj_z']: z->x, so we need z's projected features
        # mailbox['e']: z->x has a e for attention calculation
        if self.gate:       # GaAN
            return {'proj_z': edges.src['proj_z'], 'e': edges.data['e'], 'v_g': edges.src['v']}
        else:
            if self.att:    # GAT
                return {'proj_z': edges.src['proj_z'], 'e': edges.data['e']}
            else:           # GCN
                return {'proj_z': edges.src['proj_z'], 'dg_f_comb': edges.src['dg_f'] * edges.dst['dg_f']}

    def reduce_func(self, nodes):
        """ Specify how messages are processed and propagated to nodes """
        if self.att:    # GaAN & GAT
            # Aggregate features to nodes
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            alpha = F.dropout(alpha, 0.1)
            h = torch.sum(alpha * nodes.mailbox['proj_z'], dim=1)
        else:           # GCN
            h = torch.sum(nodes.mailbox['dg_f_comb'] * nodes.mailbox['proj_z'], dim=1)

        # head gates
        if self.gate:   # GaAN
            pwFeat = nodes.mailbox['v_g']
            gateProj = self.Wgm(pwFeat)
            maxFeat = torch.max(gateProj, dim=1)[0]
            meanFeat = torch.mean(pwFeat, dim=1)
            gFCVal = self.gate_fc_l(nodes.data['v']) + self.gate_fc_m(maxFeat) + self.gate_fc_r(meanFeat)
            gVal = torch.sigmoid(gFCVal)
            h = gVal * h

        return {'h': h}

    def forward(self, g: dgl.DGLGraph):
        with g.local_scope():
            if self.att:    # GAT & GaAN
                feat = g.ndata['v']

                # Wa: shared attention to features v (or h for multiple GAT layers)
                z = self.Wa(feat)
                g.ndata['z'] = z

                # AttentionNet
                g.apply_edges(self.edge_attention)
            else:           # GCN
                g.ndata['dg_f'] = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).reshape(-1, 1)

            # Message Passing
            g.update_all(self.message_func, self.reduce_func)
            res = g.ndata['proj_z'] + g.ndata['h']
            return res


class MultiHeadCGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 att=True, num_heads=Config.NUM_HEADS_DEFAULT, merge=Config.MERGE_HEAD_MODE_DEFAULT, gate=True):
        super(MultiHeadCGaANLayer, self).__init__()
        self.att = att
        self.gate = gate
        if (not self.att) and self.gate:
            sys.stderr.write('[MultiHeadPwGaANLayer] Use Attention = %s but Use Gate = %s!\n' % (str(self.att), str(self.gate)))
            exit(-985)

        self.num_heads = num_heads
        self.cGaANs = nn.ModuleList([
            CGaANLayer(in_dim, out_dim, att=self.att, gate=self.gate) for _ in range(self.num_heads)
        ])

        self.merge = merge

    def forward(self, g: dgl.DGLGraph):
        head_outs = torch.stack([self.cGaANs[i](g) for i in range(len(self.cGaANs))])
        if self.merge == 'cat':
            return head_outs.permute(1, 0, 2).reshape(head_outs.shape[-2], -1)
        elif self.merge == 'mean':
            return torch.mean(head_outs, dim=0)
        else:
            return head_outs.permute(1, 0, 2).reshape(head_outs.shape[-2], -1)  # Default: cat
