import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.dim = args.dim

        self.ent_embed = nn.Embedding(self.n_ent, self.dim)
        self.rel_embed = nn.Embedding(self.n_rel, self.dim)
        self.input_drop = nn.Dropout(args.input_drop)

        nn.init.xavier_normal_(self.ent_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)

    def forward(self, h, r, t):
        # h*r*t = t*r*h
        r = torch.cat([
            self.rel_embed.weight,
            self.rel_embed.weight
        ])[r]
        h = self.ent_embed.weight[h]
        t = self.ent_embed.weight[t]
        r = self.input_drop(r)
        h = self.input_drop(h)
        t = self.input_drop(t)
        score = torch.sum(h*r*t, dim=-1)
        return score
