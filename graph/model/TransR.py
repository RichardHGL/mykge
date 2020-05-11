import torch
import torch.nn as nn
import torch.nn.functional as F


class TransR(nn.Module):
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
        self.mat_embed = nn.Embedding(self.n_rel, self.dim * self.dim)

        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        nn.init.xavier_uniform_(self.mat_embed.weight)

    def _transfer(self, e, mat):
        e = e.unsqueeze(-2)
        if len(mat.size()) == 2:
            mat = mat.view(-1, self.dim, self.dim)
        else:
            mat = mat.view(mat.size(0), mat.size(1), self.dim, self.dim)
        e = torch.matmul(e, mat).squeeze(-2)
        return F.normalize(e, p=2, dim=-1)

    def forward(self, h, r, t):
        mat = torch.cat([
            self.mat_embed.weight,
            self.mat_embed.weight
        ])[r]
        r = torch.cat([
            self.rel_embed.weight,
            -self.rel_embed.weight
        ])[r]
        h = self.ent_embed.weight[h]
        t = self.ent_embed.weight[t]
        h = F.normalize(h, p=2, dim=-1)
        r = F.normalize(r, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        h = self._transfer(h, mat)
        t = self._transfer(t, mat)
        dist = h + r - t
        score = self.gamma - torch.norm(dist, p=self.p, dim=-1)
        return score
'''
    def regLoss(self, h, r, t):
        def norm(e):
            e = torch.sum(e**2, dim=1)
            return torch.sum(torch.max(e - torch.ones_like(e), torch.zeros_like(e)))
        r_matrix = self.transfer_matrix.weight[r]
        h = self.ent_embedding.weight[h]
        r = self.rel_embedding.weight[r]
        t = self.ent_embedding.weight[t]

        h = self._transfer(h, r_matrix)
        t = self._transfer(t, r_matrix)
        loss = norm(h) + norm(r) + norm(t)
        loss = 0
        #loss = nn.MarginRankingLoss(0., reduction='sum')(h, torch.ones_like(h), -torch.ones_like(h))
        return loss
    
    def build_candid(self, r):
        r_matrix = torch.cat([self.transfer_matrix.weight, self.transfer_matrix.weight], dim=0)[r]
        r_matrix.expand(self.args.n_entity, -1)
        return self._transfer(self.ent_embedding.weight, r_matrix)

    def evaluate(self, h, r, candid):
        h = candid[h]
        r = torch.cat([self.rel_embedding.weight, -self.rel_embedding.weight], dim=0)[r]
        
        h = F.normalize(h, p=2, dim=-1)
        r = F.normalize(h, p=2, dim=-1)
        candid = F.normalize(candid, p=2, dim=-1)
        
        hr = h + r
        hr = hr.view(-1, 1, self.args.dim_r) 
        dist = hr - candid
        scores = torch.norm(dist, p=self.args.p_norm, dim=2)
        return scores
        '''