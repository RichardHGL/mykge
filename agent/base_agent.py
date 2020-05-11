import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import time
import numpy as np

from dataset.dataloader import Dataset_A, Dataset_B, Dataset_D, EvalIterator
from graph.model import TransE, TransH, TransR, TransD
from graph.model import DistMult, ComplEx, ConvE, ConvTransE
from evaluation.eval import Evaluator
from util.utils import create_logger, file2_triple, triple2_dict


class BaseAgent():
    def __init__(self, args):
        self.args = args
        self.name_exp()
        self.logger = create_logger(self.args)
        self.best_performace = 0
        self.data_load()
        self.model_def()
        self.train_def()
        self.optim_def()
        self.evaluator_def()

    def name_exp(self):
        self.args.exp_path = os.path.join(self.args.exp_path, self.args.data)
        self.args.exp_name = '{}-{}-{}-{}'.format(
            self.args.model, self.args.data,
            self.args.mode, self.args.v
        )

    def data_load(self):
        datapath = os.path.join(self.args.path, self.args.data)
        triple_list, self.n_ent, self.n_rel = file2_triple(datapath)
        self.args.n_ent = self.n_ent
        self.args.n_rel = self.n_rel
        h_rt, r_ht = triple2_dict(triple_list, self.n_rel)
        if self.args.mode in {'1all', 'kall'}:
            dataset = Dataset_B(
                self.n_ent,
                self.n_rel,
                h_rt['train'],
                self.args.mode,
            )
        elif self.args.mode == '1vsN':
            dataset = Dataset_D(
                triple_list['train'],
                h_rt['train'],
                r_ht['train'],
                self.args
            )
        else:
            dataset = Dataset_A(
                triple_list['train'],
                h_rt['train'],
                r_ht['train'],
                self.args
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
        self.h_rt = h_rt

    def model_def(self):
        model_dict = {
            'TransE': TransE,
            'TransH': TransH,
            'TransR': TransR,
            'TransD': TransD,
            'DistMult': DistMult,
            'ComplEx': ComplEx,
            'ConvE': ConvE,
            'ConvTransE': ConvTransE
        }
        self.model = model_dict[self.args.model](self.args)
        self.model.cuda()

    def train_def(self):
        if self.args.mode in {'1all', 'kall'}:
            self.train_one_epoch = self.train_one_epoch_B
        elif self.args.mode == '1vsN':
            self.train_one_epoch = self.train_one_epoch_D
        else:
            self.train_one_epoch = self.train_one_epoch_A

    def optim_def(self):
        if self.args.optim == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr)
        else:
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr)

    def evaluator_def(self):
        self.evaluator = Evaluator(
            self.args, self.model, self.logger, self.h_rt['train'])

    def train(self):
        if self.args.trained is True:
            if self.args.pretrained_name is not None:
                filename = os.path.join(
                    self.args.exp_path, 'ckpt',
                    self.args.pretrained_name + '.ckpt'
                )
            else:
                filename = os.path.join(
                    self.args.exp_path, 'ckpt',
                    '{}-{}-rand-0.ckpt'.format(self.args.model, self.args.data)
                )
            self.load_ckpt(filename)
        # self.evaluator.evaluate(self.valid_loader)
        # self.evaluator.evaluate(self.test_loader)
        reset_time = 0
        reward = 0.
        self.logger.info('Start Training...')

        for i in range(self.args.epoch):
            t_begin = time.time()
            loss, reward = self.train_one_epoch(reward)
            self.logger.info(
                '[{:.1f}] Epoch {}: loss = {:.4f}'
                .format(time.time()-t_begin, i, loss)
            )
            if (i+1) % self.args.every == 0:
                mr, mrr, hits = self.evaluator.evaluate(self.valid_loader)
                self.evaluator.evaluate(self.test_loader)
                if mrr > self.best_performace:
                    self.best_performace = mrr
                    reset_time = 0
                    self.logger.info('Improved and model saved')
                    self.save_ckpt()
                else:
                    reset_time += 1
                    self.logger.info(
                        'Declined in recent {} epochs'.format(reset_time)
                    )
                if reset_time == 5:
                    self.logger.info('Early stopping')
                    break
        self.logger.info('Train done')
        self.evaluate_best()

    def train_one_epoch_A(self, reward):
        self.model.train()
        model = None
        if self.args.mode == 'cach':
            model = self.model
        self.train_loader.dataset.create_sample(model)
        losses = []
        for h, r, t, n_hs, n_rs, n_ts in self.train_loader:
            self.optim.zero_grad()
            h = h.cuda()
            r = r.cuda()
            t = t.cuda()
            n_hs = n_hs.cuda()
            n_rs = n_rs.cuda()
            n_ts = n_ts.cuda()

            p_score = self.model(h, r, t)
            n_scores = torch.zeros(n_hs.size()).cuda()
            if self.args.model in {'ConvE', 'ConvTransE'}:
                for k in range(n_hs.size(1)):
                    n_scores[:, k] = self.model(n_hs[:, k], n_rs[:, k], n_ts[:, k])
            else:
                n_scores = self.model(n_hs, n_rs, n_ts)
            p_loss = -F.logsigmoid(p_score)
            if self.args.mode == 'adve':
                n_loss = -(
                    F.softmax(n_scores, dim=-1) * F.logsigmoid(-n_scores)
                    ).sum(-1)
            else:
                n_loss = -torch.mean(F.logsigmoid(-n_scores), -1)
            loss = torch.sum(p_loss + n_loss)
            '''
            loss_func = torch.nn.MarginRankingLoss(4.0, reduction='sum')
            p_scores = p_score.view(-1, 1).expand_as(n_scores)
            loss = loss_func(p_scores, n_scores, torch.ones_like(p_scores))
            '''
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args.clipping_max_value
            )
            self.optim.step()
        return np.mean(losses), reward

    def train_one_epoch_B(self, reward):
        self.model.train()
        losses = []
        for e, r, y in self.train_loader:
            self.optim.zero_grad()
            e = torch.LongTensor(e).cuda().view(-1, 1)
            r = torch.LongTensor(r).cuda().view(-1, 1)
            entity = torch.arange(self.n_ent).expand(len(e), self.n_ent)
            entity = entity.long().cuda()
            scores = self.model(e, r, entity)
            y = y.cuda()
            y = ((1.0 - self.args.label_smooth_eps) * y + 1.0 / y.size(1))
            loss = nn.BCEWithLogitsLoss()(scores, y)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args.clipping_max_value
            )
            self.optim.step()
        return np.mean(losses), reward

    def train_one_epoch_D(self, reward):
        self.model.train()
        losses = []
        self.train_loader.dataset.create_sample(model)
        for e, r, entity, y in self.train_loader:
            self.optim.zero_grad()
            e = torch.LongTensor(e).cuda().view(-1, 1)
            r = torch.LongTensor(r).cuda().view(-1, 1)
            entity = entity.long().cuda()
            scores = self.model(e, r, entity)
            y = y.cuda()
            y = ((1.0 - self.args.label_smooth_eps) * y + 1.0 / 1024)
            loss = nn.BCEWithLogitsLoss()(scores, y)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args.clipping_max_value
            )
            self.optim.step()
        return np.mean(losses), reward

    def save_ckpt(self):
        filename = os.path.join(
            self.args.exp_path, 'ckpt',
            self.args.exp_name + '.ckpt'
        )
        checkpoint = {
            'best_performance': self.best_performace,
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        # self.best_performace = checkpoint['best_performance']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate_best(self):
        filename = os.path.join(
            self.args.exp_path, 'ckpt',
            self.args.exp_name + '.ckpt'
        )
        self.load_ckpt(filename)
        self.evaluator.evaluate(self.test_loader, name='test')

    def evaluate_once(self):
        if self.args.pretrained_name is None:
            filename = os.path.join(
                self.args.exp_path, 'ckpt',
                '{}-{}-{}-0.ckpt'.format(
                    self.args.model, self.args.data, self.args.mode)
            )
        else:
            filename = os.path.join(
                self.args.exp_path, 'ckpt',
                self.args.pretrained_name + '.ckpt'
            )
        self.load_ckpt(filename)
        self.evaluator.evaluate(self.valid_loader, name='test')
