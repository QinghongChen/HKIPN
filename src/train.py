import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from model import MODEL
import logging


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info):
    logging.info("================== training MODEL ====================")
    # train_data, eval_data, test_data, n_users_entities, n_relations, triplet_sets
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    triplet_sets = data_info[5]
    kg_pos_data = data_info[6]
    kg_neg_t = data_info[7]
    model, optimizer_cf, optimizer_kg, loss_func_cf = _init_model(args, data_info)
    
    for step in range(args.n_epoch):
        if step % 2 == 0:
            # train kg
            start = 0
            while start < kg_pos_data.shape[0]:
                kg_loss = model('calc_kg_loss', *_get_kg_data(args, kg_pos_data['h'][start:start + args.kg_batch_size], kg_pos_data['r'][start:start + args.kg_batch_size], 
                                kg_pos_data['t'][start:start + args.kg_batch_size], kg_neg_t[start:start + args.kg_batch_size]))
                optimizer_kg.zero_grad()
                kg_loss.backward()
                optimizer_kg.step()
                start += args.kg_batch_size

        # train cf
        train_data = shuffle(train_data)
        start = 0
        while start  < train_data.shape[0]:
            cf_labels = _get_label(args, train_data[start:start + args.cf_batch_size]['label'].values)
            cf_scores = model('calc_cf_score', *_get_cf_data(args, train_data, triplet_sets, start, start + args.cf_batch_size))
            loss = loss_func_cf(cf_scores, cf_labels)
            optimizer_cf.zero_grad()
            loss.backward()
            optimizer_cf.step()
            start += args.cf_batch_size
        
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data, triplet_sets)
        test_auc, test_f1 = ctr_eval(args, model, test_data, triplet_sets)
        ctr_info = 'epoch %.2d    eval auc: %.4f    eval f1: %.4f   test auc: %.4f    test f1: %.4f '
        logging.info(ctr_info, step, eval_auc, eval_f1, test_auc, test_f1)
        if args.show_topk:
            topk_eval(args, model, train_data, test_data, triplet_sets)


def ctr_eval(args, model, data, triplet_sets):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        cf_labels = data[start:start + args.cf_batch_size]['label'].values
        cf_scores = model('calc_cf_score', *_get_cf_data(args, data, triplet_sets, start, start + args.cf_batch_size))
        cf_scores = cf_scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=cf_labels, y_score=cf_scores)
        auc_list.append(auc)

        cf_scores[cf_scores >= 0.5] = 1
        cf_scores[cf_scores < 0.5] = 0
        f1 = f1_score(y_true=cf_labels, y_pred=cf_scores)
        f1_list.append(f1)

        start += args.cf_batch_size
    model.train()  
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1


def topk_eval(args, model, train_data, test_data, triplet_sets):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}

    item_set = set(train_data['i'].values.tolist() + test_data['i'].values.tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()    
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.test_batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.test_batch_size] 
            input_data = _get_topk_feed_data(user, items)
            cf_scores = model('calc_cf_score', *_get_cf_data(args, input_data, triplet_sets, 0, args.test_batch_size))
            for item, score in zip(items, cf_scores):
                item_score_map[item] = score
            start += args.test_batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.test_batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            cf_scores = model('calc_cf_score' ,*_get_cf_data(args, input_data, triplet_sets, 0, args.test_batch_size))
            for item, score in zip(res_items, cf_scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))

    recall = [np.mean(recall_list[k]) for k in k_list]
    _show_topk_info(zip(k_list, recall))

    
def _init_model(args, data_info):
    n_users_entities = data_info[3]
    n_relations = data_info[4]
    model = MODEL(args, n_users_entities, n_relations)
    if args.use_cuda:
        model.cuda()
    optimizer_cf = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    optimizer_kg = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    loss_func_cf = nn.BCELoss()
    return model, optimizer_cf, optimizer_kg, loss_func_cf
    
    
def _get_cf_data(args, data, triplet_sets, start, end):
    # origin item
    users = torch.LongTensor(data[start:end]['u'].values)
    items = torch.LongTensor(data[start:end]['i'].values)
    if args.use_cuda:
        items = items.cuda()
        users = users.cuda()
    # kg propagation embeddings
    users_triplet = _get_triplet_tensor(args, data[start:end]['u'].values, triplet_sets)
    items_triplet = _get_triplet_tensor(args, data[start:end]['i'].values, triplet_sets)
    return users, items, users_triplet, items_triplet


def _get_kg_data(args, h, r, t, t_neg):
    # origin item
    h = torch.LongTensor(h.to_numpy())
    r = torch.LongTensor(r.to_numpy())
    t = torch.LongTensor(t.to_numpy())
    t_neg = torch.LongTensor(np.array(t_neg))
    if args.use_cuda:
        h = h.cuda()
        r = r.cuda()
        t = t.cuda()
        t_neg = t_neg.cuda()
    return h, r, t, t_neg


def _get_triplet_tensor(args, objs, triplet_set):
    # [h,r,t]  h: [layers, batch_size, triple_set_size]
    h,r,t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([triplet_set[obj][i][0] for obj in objs]))
        r.append(torch.LongTensor([triplet_set[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([triplet_set[obj][i][2] for obj in objs]))
        if args.use_cuda:
            h = list(map(lambda x: x.cuda(), h))
            r = list(map(lambda x: x.cuda(), r))
            t = list(map(lambda x: x.cuda(), t))
    return [h,r,t]


def _get_label(args,labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels


def _get_user_record(args, data, is_train):
    user_history_dict = dict()

    for row in data.iterrows():
        u, i, label = row[1]
        if is_train or label == 1:
            if u not in user_history_dict:
                user_history_dict[u] = set()
            user_history_dict[u].add(i)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return pd.DataFrame(res, columns=['u','i'])


def _show_topk_info(recall_zip):
    res = ""
    for i,j in recall_zip:
        res += "Recall@%d:%.4f  "%(i,j)
    logging.info(res)