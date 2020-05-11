import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.dim = args.dim

        self.ent_embed = nn.Embedding(self.n_ent, self.dim)
        self.rel_embed = nn.Embedding(2*self.n_rel, self.dim)

        self.input_drop = nn.Dropout(args.input_drop)
        self.feat_drop = nn.Dropout(args.feat_drop)
        self.hidd_drop = nn.Dropout(args.hidd_drop)
        self.bn_init = nn.BatchNorm1d(1)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(args.channels)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.Conv = nn.Conv1d(
            2, args.channels, args.kernel_size, 1, padding=args.kernel_size//2)
        self.fc = nn.Linear(args.channels*self.dim, self.dim)

        nn.init.xavier_normal_(self.ent_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)

    def forward(self, h, r, t):
        # h,r: (B,) t: (B,) or (B, N)
        h = self.ent_embed.weight[h]
        r = self.rel_embed.weight[r]

        h = h.view(h.size(0), 1, -1)
        r = r.view(h.size(0), 1, -1)
        h = self.bn_init(h)
        # (B, 1, dim)
        x = torch.cat([h, r], dim=1)
        # (B, 2, dim)
        x = self.bn0(x)
        x = self.input_drop(x)
        x = self.Conv(x)
        # (B, channels, dim)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop(x)
        x = x.view(x.size(0), -1)
        # (B, channels*dim)
        x = self.fc(x)
        # (B, dim)
        x = self.hidd_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        if len(t.size()) != 1:
            score = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
        else:
            t = self.ent_embed.weight[t]
            score = torch.sum(x*t, dim=-1)
        return score
