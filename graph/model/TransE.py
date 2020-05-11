import torch
import torch.nn as nn


class TransE(nn.Module):
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
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def forward(self, h, r, t):
        h = self.ent_embed.weight[h]
        r = torch.cat([
            self.rel_embed.weight,
            -self.rel_embed.weight
        ])[r]
        t = self.ent_embed.weight[t]
        dist = h + r - t
        score = self.gamma - torch.norm(dist, p=self.p, dim=-1)
        return score

    """
    def regLoss(self, h, r, t):
        def norm(e):
            e = torch.sum(e**2, dim=1)
            return torch.sum(torch.max(e - torch.ones_like(e), torch.zeros_like(e)))
        h = self.ent_embed.weight[h]
        r = self.rel_embed.weight[r]
        t = self.ent_embed.weight[t]
        loss = norm(h) + norm(r) + norm(t)
        #loss = nn.MarginRankingLoss(0., reduction='sum')(h, torch.ones_like(h), -torch.ones_like(h))
        return loss

    def forward2(self, h, r, t):
        # for kbgan
        h = self.ent_embed.weight[h]
        r = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)[r]
        t = self.ent_embed.weight[t]
        dist = h + r - t
        score = torch.norm(dist, p=self.args.p_norm, dim=1)
        return score

    def build_candid(self, r):
        return self.ent_embed.weight

    def evaluate(self, h, r, candid):
        h = candid[h]
        r = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)[r]
        hr = h + r
        hr = hr.unsqueeze(1)  # (B*1*dim)
        dist = hr - candid  # (B*N*dim)
        scores = self.args.gamma - torch.norm(dist, p=1, dim=2)
        # scores = torch.sum(torch.abs(dist), dim=2)
        return scores
    """
