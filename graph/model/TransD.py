import torch
import torch.nn as nn
import torch.nn.functional as F


class TransD(nn.Module):
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
        self.proj_ent_embed = nn.Embedding(self.n_ent, self.dim)
        self.proj_rel_embed = nn.Embedding(self.n_rel, self.dim)

        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        nn.init.xavier_uniform_(self.proj_ent_embed.weight)
        nn.init.xavier_uniform_(self.proj_rel_embed.weight)

    def _transfer(self, e, e_t, r_t):
        return F.normalize(e + torch.sum(e*e_t, -1, True)*r_t, 2, -1)

    def forward(self, h, r, t):
        r_t = torch.cat([
            self.proj_rel_embed.weight,
            self.proj_rel_embed.weight
        ])[r]
        h_t = self.proj_ent_embed.weight[h]
        t_t = self.proj_ent_embed.weight[t]

        r = torch.cat([
            self.rel_embed.weight,
            -self.rel_embed.weight
        ])[r]
        h = self.ent_embed.weight[h]
        t = self.ent_embed.weight[t]
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)
        r = F.normalize(r, 2, -1)

        h = self._transfer(h, h_t, r_t)
        t = self._transfer(t, t_t, r_t)
        dist = h + r - t
        score = self.gamma - torch.norm(dist, p=self.p, dim=-1)
        return score
'''
    def regLoss(self, h, mode='h'):
        if mode == 'r': 
            h = self.rel_embedding.weight[h]
        else: h = self.ent_embedding.weight[h]
        h = torch.sum(h**2, dim=1)
        loss = torch.sum(torch.max(h - torch.ones_like(h), torch.zeros_like(h)))
        #loss = nn.MarginRankingLoss(0., reduction='sum')(h, torch.ones_like(h), -torch.ones_like(h))
        return loss
    
    def build_candid(self, r):
        r_transfer = torch.cat([self.rel_transfer.weight, self.rel_transfer.weight], dim=0)[r]
        r_transfer.expand(self.args.n_entity, -1)
        return self._transfer(self.ent_embedding.weight, self.ent_transfer.weight, r_transfer)

    def evaluate(self, h, r, candid):
        h = candid[h]
        r = torch.cat([self.rel_embedding.weight, -self.rel_embedding.weight], dim=0)[r]
        
        h = F.normalize(h, p=2, dim=-1)
        r = F.normalize(h, p=2, dim=-1)
        candid = F.normalize(self.candid, p=2, dim=-1)
        

        hr = h + r
        hr = hr.view(-1, 1, self.args.dim_r) 
        dist = hr - candid
        scores = torch.norm(dist, p=self.args.p_norm, dim=2)
        return scores
        '''