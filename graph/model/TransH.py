import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.dim = args.dim
        self.p = args.p
        self.gamma = args.gamma

        self.ent_embed = nn.Embedding(self.n_ent, self.dim)
        self.rel_embed = nn.Embedding(self.n_rel, self.dim)
        self.w_embed = nn.Embedding(self.n_rel, self.dim)
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        nn.init.xavier_uniform_(self.w_embed.weight)

    def _transfer(self, e, w):
        w = F.normalize(w, p=2, dim=-1)
        return e - torch.sum(e*w, -1, True)*w

    def forward(self, h, r, t):
        w = torch.cat([
            self.w_embed.weight,
            self.w_embed.weight
        ])[r]
        r = torch.cat([
            self.rel_embed.weight,
            -self.rel_embed.weight
        ])[r]
        h = self.ent_embed.weight[h]
        t = self.ent_embed.weight[t]
        h = self._transfer(h, w)
        t = self._transfer(t, w)
        dist = h + r - t
        score = self.gamma - torch.norm(dist, p=self.p, dim=-1)
        return score

