import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.dim = args.dim

        assert(self.dim == 200)
        self.ent_embed = nn.Embedding(self.n_ent, self.dim)
        self.rel_embed = nn.Embedding(2*self.n_rel, self.dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.n_ent)))

        self.input_drop = nn.Dropout(args.input_drop)
        self.feat_drop = nn.Dropout2d(args.feat_drop)
        # 2d randomly zeros entire channels
        self.hidd_drop = nn.Dropout(args.hidd_drop)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(1)

        self.Conv = nn.Conv2d(1, 32, (3, 3), bias=False)
        self.fc = nn.Linear(32*18*18, self.dim)

        nn.init.xavier_normal_(self.ent_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)

    def forward(self, h, r, t):
        # h,r: (B,) t: (B,) or (B, N)
        h = self.ent_embed.weight[h]
        r = self.rel_embed.weight[r]

        h = h.view(h.size(0), 1, 10, 20)
        r = r.view(h.size(0), 1, 10, 20)
        x = torch.cat([h, r], dim=-2)
        # (B, 1, 20, 20)
        x = self.bn0(x)
        x = self.input_drop(x)
        x = self.Conv(x)
        # (B, 32, 18, 18)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop(x)
        x = x.view(x.size(0), 1, 10368)
        # (B, 1, 32*18*18=10368)
        x = self.fc(x)
        # (B, 1, 200)
        x = self.hidd_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.squeeze(1)
        # (B, 200)

        if len(t.size()) != 1:
            b = self.b.data[t]
            t = self.ent_embed.weight[t]
            x = x.view(x.size(0), 1, -1)
            score = torch.sum(x*t, dim=-1)
        else:
            b = self.b.data[t]
            t = self.ent_embed.weight[t]
            score = torch.sum(x*t, dim=-1)
        score += b.expand_as(score)
        return score
