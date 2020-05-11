import torch
import torch.nn as nn


class ComplEx(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.dim = args.dim

        self.ent_real = nn.Embedding(self.n_ent, self.dim)
        self.ent_img = nn.Embedding(self.n_ent, self.dim)
        self.rel_real = nn.Embedding(self.n_rel, self.dim)
        self.rel_img = nn.Embedding(self.n_rel, self.dim)
        self.input_drop = nn.Dropout(args.input_drop)

        nn.init.xavier_normal_(self.ent_real.weight)
        nn.init.xavier_normal_(self.ent_img.weight)
        nn.init.xavier_normal_(self.rel_real.weight)
        nn.init.xavier_normal_(self.rel_img.weight)

    def forward(self, h, r, t):
        # Re(h*r*t_) = Re(t*r_*h_)
        r2 = torch.cat([
            self.rel_real.weight,
            self.rel_real.weight
        ])[r]
        i2 = torch.cat([
            self.rel_img.weight,
            -self.rel_img.weight
        ])[r]
        r1 = self.ent_real.weight[h]
        i1 = self.ent_img.weight[h]
        r3 = self.ent_real.weight[t]
        i3 = self.ent_img.weight[t]

        r1 = self.input_drop(r1)
        r2 = self.input_drop(r2)
        r3 = self.input_drop(r3)
        i1 = self.input_drop(i1)
        i2 = self.input_drop(i2)
        i3 = self.input_drop(i3)

        rrr = torch.sum(r1*r2*r3, dim=-1)
        iri = torch.sum(i1*r2*i3, dim=-1)
        rii = torch.sum(r1*i2*i3, dim=-1)
        iir = torch.sum(i1*i2*r3, dim=-1)
        score = rrr + iri + rii - iir
        score = score
        return score
