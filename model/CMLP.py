import torch
import torch.nn as nn
import dgl

import Config


def init_weights(layer, gain):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight, gain=gain)


class CMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim_ref, out_dim):
        super(CMLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim_ref = hidden_dim_ref
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim_ref),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_ref, int(2 * self.hidden_dim_ref)),
            nn.ReLU(),
            nn.Linear(int(2 * self.hidden_dim_ref), self.hidden_dim_ref),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_ref, self.out_dim),
            nn.ReLU()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            init_weights(layer, gain)

    def forward(self, _, feat: torch.Tensor):   # Unify number of arguments by setting a "_"
        out = self.layers(feat)
        return out


if __name__ == '__main__':
    """ Test """
    mlp = CMLP(in_dim=Config.FEAT_DIM_DEFAULT, hidden_dim_ref=Config.HIDDEN_DIM_DEFAULT, out_dim=Config.NUM_CLASSES)
    (graph,), _ = dgl.load_graphs('../data/cora/cora.dgl')
    feats = graph.ndata['feat']
    res = mlp(None, feats)
    print(res.shape)
