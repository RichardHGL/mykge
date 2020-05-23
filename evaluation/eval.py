import torch
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from util.utils import write_record, eval_ranks


class Evaluator:
    def __init__(self, args, model, logger, h_rt):
        self.args = args
        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.model = model
        self.filename = os.path.join(
            self.args.exp_path, 'txt', self.args.exp_name + '.txt')
        self.logger = logger

        self.h_rt = h_rt
        self.init_stat()

    def evaluate(self, data_loader, name='valid'):
        demu = 0
        dezi = 0
        self.model.eval()
        self.logger.info('Start Evaling...')

        head_rank = []
        tail_rank = []
        # [raw_tail, fil_tail raw_head, fil_head]
        record = defaultdict(lambda: [0, 0, 0, 0])
        pbar = tqdm(total=len(data_loader))
        for head, rel, tails, all_tails in data_loader:
            head = torch.LongTensor(head).cuda().view(-1, 1)
            rel = torch.LongTensor(rel).cuda().view(-1, 1)
            entity = torch.arange(self.n_ent).expand(len(head), self.n_ent)
            entity = entity.long().cuda()
            scores = self.model(head, rel, entity)
            for i in range(len(head)):
                _, ind1 = torch.sort(scores[i], descending=True)
                ind1 = ind1.cpu().numpy()
                for t in tails[i]:
                    raw_rank = np.where(ind1 == t)[0][0] + 1
                    # filter rank
                    temp1 = scores[i][t].item()
                    filts = all_tails[i]
                    temp2 = scores[i][filts]
                    scores[i][filts] = -float('inf')
                    scores[i][t] = temp1
                    _, ind2 = torch.sort(scores[i], descending=True)
                    ind2 = ind2.cpu().numpy()
                    fil_rank = np.where(ind2 == t)[0][0] + 1
                    scores[i][filts] = temp2

                    h = head[i].cpu().item()
                    r = rel[i].cpu().item()
                    flag = False
                    if fil_rank <= 10:
                        flag = True
                        if fil_rank != 1:
                            dezi += 1
                    demu += 1
                    mask = self.add_stat(ind2[: min(2, fil_rank-1)], h, r, t, flag)
                    fil_rank -= 0
                    if r < self.n_rel:
                        tail_rank.append(fil_rank)
                        record[(h, r, t)][0] = raw_rank
                        record[(h, r, t)][1] = fil_rank
                    else:
                        r = r - self.n_rel
                        head_rank.append(fil_rank)
                        record[(t, r, h)][2] = raw_rank
                        record[(t, r, h)][3] = fil_rank
            pbar.update(1)
        pbar.close()
        head_metric, tail_metric, all_metric = eval_ranks(head_rank, tail_rank)
        self.logger.info(
            'Head: mr:{:.1f} mrr:{:.2f} hits10:{:.2f} hits3:{:.2f} hits1:{:.2f}'.format
            (head_metric[0], head_metric[1], head_metric[2], head_metric[3], head_metric[4]))
        self.logger.info(
            'Tail: mr:{:.1f} mrr:{:.2f} hits10:{:.2f} hits3:{:.2f} hits1:{:.2f}'.format
            (tail_metric[0], tail_metric[1], tail_metric[2], tail_metric[3], tail_metric[4]))
        self.logger.info(
            'All: mr:{:.1f} mrr:{:.2f} hits10:{:.2f} hits3:{:.2f} hits1:{:.2f}'.format
            (all_metric[0], all_metric[1], all_metric[2], all_metric[3], all_metric[4]))
        if name == 'test':
            write_record(record, self.filename)

        print('prop: %.4f' % (dezi / demu))
        self.print_stat()
        return all_metric

    def init_stat(self):
        self.neb, self.neb_r, self.r_neb = 0, 0, 0
        self.t1, self.t2, self.t3 = 0, 0, 0
        self.union = 0
        self.un_t = 0

        self.totoal = 0
        self.n_test = 0

        self.ent_nebs = dict()
        for h in self.h_rt:
            self.ent_nebs[h] = set()
            for r in self.h_rt[h]:
                self.ent_nebs[h] |= self.h_rt[h][r]

    def add_stat(self, indx, h, r, t, flag):
        neb, neb_r, r_neb = set(), set(), set()

        if h in self.ent_nebs:
            neb = self.ent_nebs[h]
        for ent in neb:
            if ent in self.h_rt:
                if r in self.h_rt[ent]:
                    neb_r |= self.h_rt[ent][r]
        if h in self.h_rt:
            if r in self.h_rt[h]:
                for ent in self.h_rt[h][r]:
                    r_neb |= self.ent_nebs[ent]
        union = neb | r_neb | neb_r

        if flag is True:
            self.totoal += len(indx)
            for s in indx:
                if s in neb:
                    self.neb += 1
                if s in neb_r:
                    self.neb_r += 1
                if s in r_neb:
                    self.r_neb += 1
                if s in union:
                    self.union += 1

        self .n_test += 1
        if t in neb:
            self.t1 += 1
        if t in neb_r:
            self.t2 += 1
        if t in r_neb:
            self.t3 += 1
        if t in union:
            self.un_t += 1

        mask = 0
        for s in indx:
            if s in (neb_r | r_neb):
                mask += 1
        return mask

    def print_stat(self):
        neb = self.neb / self.totoal
        neb_r = self.neb_r / self.totoal
        r_neb = self.r_neb / self.totoal
        union = self.union / self.totoal

        print('%.4f, %.4f, %.4f, %.4f' % (neb, neb_r, r_neb, union))
        print('%.4f, %.4f, %.4f, %.4f' % (
            self.t1/self.n_test, self.t2/self.n_test, self.t3/self.n_test, self.un_t/self.n_test))
