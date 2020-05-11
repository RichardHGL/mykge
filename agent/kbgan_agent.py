import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np

from .base_agent import BaseAgent
from dataset.dataloader import Dataset_C, EvalIterator
from graph.model import TransH, DistMult
from evaluation.eval import Evaluator
from util.utils import file2_triple, triple2_dict


class KbganAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        if args.scratch is False:
            self.D_G_init()

    def data_load(self):
        datapath = os.path.join(self.args.path, self.args.data)
        triple_list, self.n_ent, self.n_rel = file2_triple(datapath)
        self.args.n_ent = self.n_ent
        self.args.n_rel = self.n_rel
        h_rt, r_ht = triple2_dict(triple_list, self.n_rel)
        dataset = Dataset_C(
            self.n_ent,
            self.n_rel,
            triple_list['train'],
            h_rt['train'],
            self.args.bern,
            self.args.n_pool
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.valid_loader = EvalIterator(
            h_rt['valid'], h_rt['all'], self.args.batch_size2
        )
        self.test_loader = EvalIterator(
            h_rt['test'], h_rt['all'], self.args.batch_size2
        )

        self.logger.info('Dataload done')
        self.logger.info('-'*38)
        self.logger.info('N_ENTITY: {}'.format(self.n_ent))
        self.logger.info('N_RELATION: {}'.format(self.n_rel))
        self.logger.info('-'*38)
        self.h_rt = h_rt['train']

    def model_def(self):
        self.D = TransH(self.args).cuda()
        self.G = DistMult(self.args).cuda()

    def train_def(self):
        self.train_one_epoch = self.train_one_epoch_C

    def optim_def(self):
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr)
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr)

    def evaluator_def(self):
        self.evaluator = Evaluator(self.args, self.D, self.logger, self.h_rt)

    def D_G_init(self):
        if self.args.D_file is None:
            self.args.D_file = '{}-{}-rand-0'.format(
                self.args.model_D, self.args.data
            )
        if self.args.G_file is None:
            self.args.G_file = '{}-{}-rand-0'.format(
                self.args.model_G, self.args.data
            )
        file_D = os.path.join(
            self.args.exp_path, 'ckpt', self.args.D_file + '.ckpt'
        )
        file_G = os.path.join(
            self.args.exp_path, 'ckpt', self.args.G_file + '.ckpt'
        )
        self.D.load_state_dict(torch.load(file_D)['model_state_dict'])
        self.G.load_state_dict(torch.load(file_G)['model_state_dict'])
        self.logger.info('Pretrained D_G load done')

    def train_one_epoch_C(self, b):
        self.D.train()
        self.G.train()
        self.train_loader.dataset.create_sample()
        losses = []
        rewards = []
        for h, r, t, n_t in self.train_loader:
            h = h.cuda()
            r = r.cuda()
            t = t.cuda()
            n_t = n_t.cuda()
            G_step = self.G_step(h, r, n_t, self.args.n_sample)
            # G: generate n_h, n_t
            n_t = next(G_step)
            # D: train D with n_h, n_t
            self.optim_D.zero_grad()
            p_score = self.D(h, r, t)
            h = h.view(-1, 1)
            r = r.view(-1, 1)
            n_score = self.D(h, r, n_t)
            p_loss = -F.logsigmoid(p_score)
            n_loss = -torch.mean(F.logsigmoid(-n_score), -1)
            loss = torch.sum(p_loss + n_loss)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(
                [param for _, param in self.D.named_parameters()],
                self.args.clipping_max_value
            )
            self.optim_D.step()
            reward = n_score.detach()
            rewards.append(torch.sum(n_score).item())
            G_step.send(reward - b)
            # G: train D with reward
        return np.mean(losses), np.mean(rewards)

    def G_step(self, h, r, n_t, n_sample, temp=1.0):
        # G: generate n_t
        h = h.view(-1, 1)
        r = r.view(-1, 1)
        score = self.G(h, r, n_t) / temp
        prob = F.softmax(score, dim=-1)
        cols = torch.multinomial(prob, n_sample, replacement=True)
        rows = torch.arange(h.size(0)).view(-1, 1).long()
        n_t = n_t[rows, cols]
        reward = yield n_t

        # G: train D with reward
        self.optim_G.zero_grad()
        log_prob = F.log_softmax(score, dim=-1)
        loss = -torch.sum(reward*log_prob[rows, cols])
        loss.backward()
        nn.utils.clip_grad_norm_(
            [param for _, param in self.G.named_parameters()],
            self.args.clipping_max_value)
        self.optim_G.step()
        yield None

    def save_ckpt(self):
        filename = os.path.join(
            self.args.exp_path, 'ckpt',
            self.args.exp_name + '.ckpt'
        )
        checkpoint = {
            'best_performance': self.best_performace,
            'D_state_dict': self.D.state_dict(),
            'G_state_dict': self.G.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        self.best_performace = checkpoint['best_performance']
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G.load_state_dict(checkpoint['G_state_dict'])

    def evaluate_best(self):
        filename = os.path.join(
            self.args.exp_path, 'ckpt',
            self.args.exp_name + '.ckpt'
        )
        self.load_ckpt(filename)
        self.evaluator.evaluate(self.test_loader, name='test')

    def evaluate_once(self):
        filename = os.path.join(
            self.args.exp_path, 'ckpt',
            self.args.exp_name + '.ckpt'
        )
        self.load_ckpt(filename)
        self.Evaluator.evaluate(self.test_loader, name='test')
