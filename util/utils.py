import logging
import os
import numpy as np
from collections import defaultdict


def create_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(formatter)
    if args.f is True:
        logger.addHandler(handler1)

    if args.eval is False:
        handler2 = logging.FileHandler(
            filename=os.path.join(args.exp_path, 'log', args.exp_name + '.log',),
            mode='w')
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)

    for name, value in args.__dict__.items():
        logger.info('{}: {}'.format(name.upper(), str(value)))
    logger.info('-'*38)
    return logger


# agent
def file2_triple(path):
    triple_list = {}
    triple_list['all'] = []
    for name in ['train', 'valid', 'test']:
        lines = open(os.path.join(path, name+'.txt'), 'r').readlines()
        triple_list[name] = []
        for l in lines:
            # OpenKE triplet is (e1,e2,rel)
            e1, e2, rel = l.strip().split(' ')
            h, r, t = int(e1), int(rel), int(e2)
            triple_list[name].append([h, r, t])
        triple_list['all'].extend(triple_list[name])
    n_entity = max(
        max(np.array(triple_list['all'])[:, 0]),
        max(np.array(triple_list['all'])[:, -1])) + 1
    n_relation = max(np.array(triple_list['all'])[:, 1]) + 1
    return triple_list, n_entity, n_relation


def triple2_dict(triple_list, n_relation):
    h_rt, r_ht = {}, {}
    for name, triples in triple_list.items():
        h_rt[name] = {}
        r_ht[name] = {}
        for h, r, t in triples:
            r_inv = r + n_relation
            # h_rt
            h_rt[name].setdefault(h, {})
            h_rt[name][h].setdefault(r, set())
            h_rt[name][h][r].add(t)
            h_rt[name].setdefault(t, {})
            h_rt[name][t].setdefault(r_inv, set())
            h_rt[name][t][r_inv].add(h)
            # r_ht
            r_ht[name].setdefault(r, {})
            r_ht[name][r].setdefault(h, set())
            r_ht[name][r][h].add(t)
            r_ht[name].setdefault(r_inv, {})
            r_ht[name][r_inv].setdefault(t, set())
            r_ht[name][r_inv][t].add(h)
    return h_rt, r_ht


# dataset
def get_bern_prob(n_rel, h_rt, bern):
    rel_h = defaultdict(lambda: set())
    rel_t = defaultdict(lambda: set())
    for h in h_rt:
        for r in h_rt[h]:
            for t in h_rt[h][r]:
                if (r >= n_rel):
                    continue
                rel_h[r].add(h)
                rel_t[r].add(t)
    bern_prob = np.zeros(2 * n_rel)
    for k in rel_h.keys():
        left = 1 / len(rel_h[k])
        right = 1 / len(rel_t[k])
        bern_prob[k] = left / (left + right)
        bern_prob[k + n_rel] = 1 - bern_prob[k]
    if (bern is False):
        bern_prob = np.ones(2 * n_rel) / 2
    return bern_prob


# evaluation
def eval_ranks(head_rank, tail_rank):
    def calc(rank):
        rank = np.array(rank)
        mr = np.mean(rank)
        mrr = np.mean(1.0 / rank) * 100
        hits10 = np.mean(rank <= 10) * 100
        hits3 = np.mean(rank <= 3) * 100
        hits1 = np.mean(rank <= 1) * 100
        return [mr, mrr, hits10, hits3, hits1]
    head_metric = calc(head_rank)
    tail_metric = calc(tail_rank)
    all_metric = calc(head_rank + tail_rank)
    return head_metric, tail_metric, all_metric


def write_record(record, filename):
    f = open(filename, 'w')
    for hrt, rank in record.items():
        h, r, t = hrt
        raw_tail, fil_tail, raw_head, fil_head = rank
        f.write(
            '{} {} {} {} {} {} {}\n'.format
            (h, r, t, raw_tail, fil_tail, raw_head, fil_head))
    f.close()
