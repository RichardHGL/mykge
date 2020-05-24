import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from util.utils import get_bern_prob


class Dataset_A(Dataset):
    def __init__(self, data, h_rt, r_ht, args):
        super().__init__()
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.h_rt = h_rt
        self.r_ht = r_ht
        self.data = data
        self.n_sample = args.n_sample
        self.mode = args.mode
        self.bern = args.bern
        self.bern_prob = get_bern_prob(self.n_rel, h_rt, self.bern)

        if self.mode == 'popu':
            self.popu_init()
        if self.mode == 'corr':
            self.corr_init()
        if self.mode == 'cach':
            self.cach_init()
        if self.mode == 'grap':
            self.grap_init()
            self.sample_rate = args.rate

        sample_dict = {
            'rand': self.my_rand,
            'rela': self.rela,
            'popu': self.popu,
            'corr': self.corr,
            'adve': self.my_rand,
            'cach': self.cach,
            'grap': self.grap,
            }
        self.sample = sample_dict[self.mode]

    def create_sample(self, model=None):
        self.model = model
        self.samples = self.sample()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, r, t, n_hs, n_rs, n_ts = self.samples[idx]
        n_hs = torch.LongTensor(n_hs)
        n_rs = torch.LongTensor(n_rs)
        n_ts = torch.LongTensor(n_ts)
        return h, r, t, n_hs, n_rs, n_ts

    def my_rand(self):
        samples = []
        n = len(self.data)
        negs_h = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        negs_t = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        for i in range(n):
            h, r, t = self.data[i]
            n_h = negs_h[i]
            n_r = [r] * self.n_sample
            n_t = negs_t[i]
            select = np.random.binomial(1, self.bern_prob[n_r])
            for k, sel in enumerate(select):
                if (sel == 1):
                    n_t[k] = t  # replace head
                else:
                    n_h[k] = h
            samples.append([h, r, t, n_h, n_r, n_t])
        return samples

    def rela(self):
        samples = []
        n = len(self.data)
        negs_h = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        negs_t = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        negs_r = np.random.randint(
            low=0, high=self.n_rel, size=(n, self.n_sample))
        for i in range(n):
            h, r, t = self.data[i]
            n_h = negs_h[i]
            n_r = negs_r[i]
            n_t = negs_t[i]
            select = np.random.binomial(1, self.bern_prob[n_r])
            for k, sel in enumerate(select):
                if (sel == 1):
                    n_t[k] = t  # replace head
                else:
                    n_h[k] = h
            samples.append([h, r, t, n_h, n_r, n_t])
        return samples

    def popu_init(self):
        self.tail_cnt = np.ones(self.n_ent)
        self.head_cnt = np.ones(self.n_ent)
        for h, r, t in self.data:
            self.tail_cnt[t] += 1
            self.head_cnt[h] += 1

    def popu(self):
        samples = []
        n = len(self.data)
        negs_h = np.random.choice(
            range(self.n_ent), size=(n, self.n_sample),
            replace=True, p=self.head_cnt / (n + self.n_ent)
        )
        negs_t = np.random.choice(
            range(self.n_ent), size=(n, self.n_sample),
            replace=True, p=self.tail_cnt / (n + self.n_ent)
        )
        for i in range(n):
            h, r, t = self.data[i]
            n_h = negs_h[i]
            n_r = [r] * self.n_sample
            n_t = negs_t[i]
            select = np.random.binomial(1, self.bern_prob[n_r])
            for k, sel in enumerate(select):
                if (sel == 1):
                    n_t[k] = t  # replace head
                else:
                    n_h[k] = h
            samples.append([h, r, t, n_h, n_r, n_t])
        return samples

    def corr_init(self):
        self.negs = []
        n = len(self.data)
        for i in range(n):
            h, r, t = self.data[i]
            neg_t = set()
            neg_h = set()
            for r_ in self.h_rt[h]:
                neg_t |= self.h_rt[h][r_]
            for r_ in self.h_rt[t]:
                neg_h |= self.h_rt[t][r_]
            neg_t -= self.h_rt[h][r]
            neg_h -= self.h_rt[t][r + self.n_rel]
            self.negs.append([list(neg_h), list(neg_t)])

    def corr(self):
        samples = []
        n = len(self.data)
        rand_h = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        rand_t = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        for i in range(n):
            h, r, t = self.data[i]
            neg_h, neg_t = self.negs[i]
            if len(neg_h) >= self.n_sample:
                n_h = np.random.choice(neg_h, self.n_sample)
            else:
                n_h = rand_h[i]
            if len(neg_t) >= self.n_sample:
                n_t = np.random.choice(neg_t, self.n_sample)
            else:
                n_t = rand_t[i]
            n_r = [r] * self.n_sample
            select = np.random.binomial(1, self.bern_prob[n_r])
            for k, sel in enumerate(select):
                if (sel == 1):
                    n_t[k] = t  # replace head
                else:
                    n_h[k] = h
            samples.append([h, r, t, n_h, n_r, n_t])
        return samples

    def cach_init(self):
        self.N1 = 30
        self.N2 = 30
        self.remove = False

        self.hr_idx = []
        self.tr_idx = []
        hr_count = 0
        tr_count = 0
        idx_dict = {}
        for h, r, t in self.data:
            if (h, r) not in idx_dict:
                idx_dict[(h, r)] = hr_count
                hr_count += 1
            if (t, r + self.n_rel) not in idx_dict:
                idx_dict[(t, r + self.n_rel)] = tr_count
                tr_count += 1
            self.hr_idx.append(idx_dict[(h, r)])
            self.tr_idx.append(idx_dict[t, r + self.n_rel])

        self.hr_cache = np.random.randint(
            low=0, high=self.n_ent, size=(hr_count, self.N1)
        )
        self.tr_cache = np.random.randint(
            low=0, high=self.n_ent, size=(tr_count, self.N1)
        )

    def cach(self):
        # update & sample
        assert(self.n_sample < self.N1)
        samples = []
        batch_size = 256
        start = 0
        end = min(start + batch_size, len(self.data))
        data = np.array(self.data)
        while (start < end):
            h = data[start: end, 0]
            r = data[start: end, 1]
            t = data[start: end, 2]
            h = torch.LongTensor(h).cuda().view(-1, 1)
            r = torch.LongTensor(r).cuda().view(-1, 1)
            t = torch.LongTensor(t).cuda().view(-1, 1)
            hr_idx = self.hr_idx[start: end]
            tr_idx = self.tr_idx[start: end]
            tail = self.hr_cache[hr_idx]
            head = self.tr_cache[tr_idx]
            # update
            tail_cand = np.random.randint(
                low=0, high=self.n_ent, size=(len(tail), self.N2))
            head_cand = np.random.randint(
                low=0, high=self.n_ent, size=(len(head), self.N2))
            tail = np.concatenate([tail, tail_cand], 1)
            head = np.concatenate([head, head_cand], 1)
            tail = torch.from_numpy(tail).type(torch.LongTensor).cuda()
            head = torch.from_numpy(head).type(torch.LongTensor).cuda()
            tail_score = self.model(h, r, tail)
            head_score = self.model(t, r + self.n_rel, head)
            tail_prob = F.softmax(tail_score, dim=-1)
            head_prob = F.softmax(head_score, dim=-1)
            tail_col = torch.multinomial(tail_prob, self.N1, replacement=True)
            head_col = torch.multinomial(head_prob, self.N1, replacement=True)
            rows = torch.arange(0, len(tail_col)).view(-1, 1).long()
            tail_rep = tail[rows, tail_col]
            head_rep = tail[rows, head_col]
            # remove (opitional)
            '''
            if self.remove is True:
                for i in range(len(h)):
                    h_ = h[i][0]
                    r_ = r[i][0]
                    t_ = t[i][0]
                    for j in range(self.N1):
                        if tail_rep[i][j] in self.h_rt[h_][r_]:
                            tail_rep[i][j] = np.random.randint(0, self.n_ent)
                        if head_rep[i][j] in self.h_rt[t_][r_ + self.n_rel]:
                            head_rep[i][j] = np.random.randint(0, self.n_ent)
                self.hr_cache[hr_idx] = tail_rep
                self.tr_cache[tr_idx] = head_rep
            '''
            # sample:
            tail_score = tail_score[rows, tail_col]
            head_score = head_score[rows, head_col]
            tail_prob = F.softmax(tail_score, dim=-1)
            head_prob = F.softmax(head_score, dim=-1)
            tail_col = torch.multinomial(tail_prob, self.n_sample, True)
            head_col = torch.multinomial(head_prob, self.n_sample, True)
            n_t = tail_rep[rows, tail_col].cpu().numpy()
            n_h = head_rep[rows, head_col].cpu().numpy()
            h = h.cpu().numpy()
            r = r.cpu().numpy()
            t = t.cpu().numpy()
            n_r = np.tile(r, (1, self.n_sample))
            # select by bernoulli
            select = np.random.binomial(1, self.bern_prob[n_r])
            for i in range(len(h)):
                for j, sel in enumerate(select[i]):
                    if sel == 1:  # replace head
                        n_t[i][j] = t[i][0]
                    else:
                        n_h[i][j] = h[i][0]
            for i in range(len(h)):
                single = [h[i][0], r[i][0], t[i][0], n_h[i], n_r[i], n_t[i]]
                samples.append(single)
            start = end
            end = min(start + batch_size, len(self.data))
        return samples

    def grap_init(self):
        n = len(self.data)
        self.hr_idx = []
        self.tr_idx = []
        hr_count = 0
        tr_count = 0
        idx_dict = {}
        for h, r, t in self.data:
            if (h, r) not in idx_dict:
                idx_dict[(h, r)] = hr_count
                hr_count += 1
            if (t, r + self.n_rel) not in idx_dict:
                idx_dict[(t, r + self.n_rel)] = tr_count
                tr_count += 1
            self.hr_idx.append(idx_dict[(h, r)])
            self.tr_idx.append(idx_dict[t, r + self.n_rel])
        self.hr_cache = []
        self.tr_cache = []
        nebs = dict()
        for h in self.h_rt:
            nebs[h] = set()
            for r in self.h_rt[h]:
                nebs[h] |= self.h_rt[h][r]

        cnt0, cnt1 = 0, 0
        for i in range(n):
            hr_idx = self.hr_idx[i]
            tr_idx = self.tr_idx[i]
            h, r, t = self.data[i]
            r_inv = r + self.n_rel
            if hr_idx == cnt0:
                cnt0 += 1
                neb, neb_r, r_neb = set(), set(), set()
                neb = nebs[h]
                for ent in neb:
                    if ent in self.h_rt:
                        if r in self.h_rt[ent]:
                            neb_r |= self.h_rt[ent][r]
                for ent in self.h_rt[h][r]:
                    r_neb |= nebs[ent]
                neg_t = ( neb | neb_r | r_neb ) - self.h_rt[h][r]
                # neg_t = neb - self.h_rt[h][r]
                if (len(neg_t) == 0):
                    neg_t = np.random.randint(0, self.n_ent, self.n_sample)
                self.hr_cache.append(list(neg_t))
            if tr_idx == cnt1:
                cnt1 += 1
                neb, neb_r, r_neb = set(), set(), set()
                neb = nebs[t]
                for ent in neb:
                    if ent in self.h_rt:
                        if r_inv in self.h_rt[ent]:
                            neb_r |= self.h_rt[ent][r_inv]
                for ent in self.h_rt[t][r_inv]:
                    r_neb |= nebs[ent]
                neg_h = ( neb | neb_r | r_neb) - self.h_rt[t][r_inv]
                # neg_h = neb - self.h_rt[t][r_inv]
                if (len(neg_h) == 0):
                    neg_h = np.random.randint(0, self.n_ent, self.n_sample)
                self.tr_cache.append(list(neg_h))

    def grap(self):
        samples = []
        n = len(self.data)
        rand_h = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        rand_t = np.random.randint(
            low=0, high=self.n_ent, size=(n, self.n_sample))
        
        '''
        hr_grap = []
        tr_grap = []
        cnt0, cnt1 = 0, 0
        for i in range(n):
            hr_idx = self.hr_idx[i]
            tr_idx = self.tr_idx[i]
            if hr_idx == cnt0:
                cnt0 += 1
                neg_t = self.hr_cache[hr_idx]
                hr_grap.append(np.random.choice(neg_t, n_sample1))
            if tr_idx == cnt1:
                cnt1 += 1
                neg_h = self.tr_cache[tr_idx]
                tr_grap.append(np.random.choice(neg_h, n_sample1))
        '''
        sb = np.random.binomial(self.n_sample, self.sample_rate, (2,n))
        for i in range(n):
            h, r, t = self.data[i]
            hr_idx = self.hr_idx[i]
            tr_idx = self.tr_idx[i]
            n_h1 = self.tr_cache[tr_idx]
            n_t1 = self.hr_cache[hr_idx]
            n_h = rand_h[i]
            n_t = rand_t[i]
            x1 = sb[0][i]
            x2 = sb[1][i]
            n_h[:x1] = np.random.choice(n_h1, x1)
            n_t[:x2] = np.random.choice(n_t1, x2)
            n_r = [r] * self.n_sample
            select = np.random.binomial(1, self.bern_prob[n_r])
            for k, sel in enumerate(select):
                if (sel == 1):
                    n_t[k] = t  # replace head
                else:
                    n_h[k] = h
            samples.append([h, r, t, n_h, n_r, n_t])
        return samples


class Dataset_B(Dataset):
    def __init__(self, n_ent, n_rel, h_rt, mode):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.h_rt = h_rt
        self.samples = []

        samples1, samples2 = [], []
        for h in h_rt:
            for r in h_rt[h]:
                tails = list(self.h_rt[h][r])
                samples2.append([h, r, tails])  # for kvsall
                for t in tails:
                    samples1.append([h, r, [t]])  # for onevsall
        if mode == '1all':
            self.samples = samples1
        else:
            self.samples = samples2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, r, tails = self.samples[idx]
        y = np.zeros(self.n_ent)
        y[tails] = 1.
        y = torch.FloatTensor(y)
        return h, r, y


class Dataset_C(Dataset):
    def __init__(self, n_ent, n_rel, data, h_rt, bern, n_pool):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.data = data
        self.n_pool = n_pool
        self.bern_prob = get_bern_prob(n_rel, h_rt, bern)

    def create_sample(self):
        samples = []
        n = len(self.data)
        negs = np.random.randint(
            0, self.n_ent, size=(2*len(self.data), self.n_pool))
        for i in range(len(self.data)):
            h, r, t = self.data[i]
            sel = np.random.binomial(1, self.bern_prob[r])
            if sel == 1:  # replace head
                samples.append([t, r + self.n_rel, h, negs[i + n]])
            else:
                samples.append([h, r, t, negs[i]])
        self.samples = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, r, t, n_t = self.samples[idx]
        return h, r, t, torch.LongTensor(n_t)


class Dataset_D(Dataset):
    def __init__(self, data, h_rt, r_ht, args):
        super().__init__()
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.h_rt = h_rt
        self.r_ht = r_ht
        self.data = data
        self.n_sample = args.n_sample

        self.grap_init()
        self.sample_rate = args.rate

    def create_sample(self):
        self.samples = self.grap()

    def __len__(self):
        return 2*len(self.data)

    def __getitem__(self, idx):
        h, r, t, neg_t = self.samples[idx]
        tails = np.append(neg_t, [t])
        y = [0]*self.n_sample + [1]
        return h, r, torch.LongTensor(tails), torch.FloatTensor(y)

    def grap_init(self):
        n = len(self.data)
        self.hr_idx = []
        self.tr_idx = []
        hr_count = 0
        tr_count = 0
        idx_dict = {}
        for h, r, t in self.data:
            if (h, r) not in idx_dict:
                idx_dict[(h, r)] = hr_count
                hr_count += 1
            if (t, r + self.n_rel) not in idx_dict:
                idx_dict[(t, r + self.n_rel)] = tr_count
                tr_count += 1
            self.hr_idx.append(idx_dict[(h, r)])
            self.tr_idx.append(idx_dict[t, r + self.n_rel])
        self.hr_cache = []
        self.tr_cache = []
        nebs = dict()
        for h in self.h_rt:
            nebs[h] = set()
            for r in self.h_rt[h]:
                nebs[h] |= self.h_rt[h][r]

        cnt0, cnt1 = 0, 0
        for i in range(n):
            hr_idx = self.hr_idx[i]
            tr_idx = self.tr_idx[i]
            h, r, t = self.data[i]
            r_inv = r + self.n_rel
            if hr_idx == cnt0:
                cnt0 += 1
                neb, neb_r, r_neb = set(), set(), set()
                neb = nebs[h]
                for ent in neb:
                    if ent in self.h_rt:
                        if r in self.h_rt[ent]:
                            neb_r |= self.h_rt[ent][r]
                for ent in self.h_rt[h][r]:
                    r_neb |= nebs[ent]
                neg_t = (neb) - self.h_rt[h][r]
                if (len(neg_t) == 0):
                    neg_t = np.random.randint(0, self.n_ent, self.n_sample)
                self.hr_cache.append(list(neg_t))
            if tr_idx == cnt1:
                cnt1 += 1
                neb, neb_r, r_neb = set(), set(), set()
                neb = nebs[t]
                for ent in neb:
                    if ent in self.h_rt:
                        if r_inv in self.h_rt[ent]:
                            neb_r |= self.h_rt[ent][r_inv]
                for ent in self.h_rt[t][r_inv]:
                    r_neb |= nebs[ent]
                neg_h = (neb) - self.h_rt[t][r_inv]
                if (len(neg_h) == 0):
                    neg_h = np.random.randint(0, self.n_ent, self.n_sample)
                self.tr_cache.append(list(neg_h))

    def grap(self):
        samples = []
        n = len(self.data)
        n_sample1 = int(self.n_sample * self.sample_rate)
        n_sample2 = self.n_sample - n_sample1
        rand_h = np.random.randint(
            low=0, high=self.n_ent, size=(n, n_sample2))
        rand_t = np.random.randint(
            low=0, high=self.n_ent, size=(n, n_sample2))

        for i in range(n):
            hr_idx = self.hr_idx[i]
            tr_idx = self.tr_idx[i]
            h, r, t = self.data[i]
            r_inv = r + self.n_rel
            
            n_t = self.hr_cache[hr_idx]
            n_t = np.append(np.random.choice(n_t, n_sample1), rand_t[i])
            samples.append([h, r, t, n_t])

            n_h = self.tr_cache[tr_idx]
            n_h = np.append(np.random.choice(n_h, n_sample1), rand_h[i])
            samples.append([t, r_inv, h, n_h])
        return samples


def EvalIterator(h_rt, h_rt_all, batch_size):
    eval_data = []
    count = 0
    head, rel, tails, all_tails = [], [], [], []
    for h in h_rt:
        for r in h_rt[h]:
            head.append(h)
            rel.append(r)
            tails.append(list(h_rt[h][r]))
            all_tails.append(list(h_rt_all[h][r]))
            count += 1
            if count % batch_size == 0:
                eval_data.append([head, rel, tails, all_tails])
                head, rel, tails, all_tails = [], [], [], []
    if count > 0:
        eval_data.append([head, rel, tails, all_tails])
    return eval_data
