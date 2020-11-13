import collections
import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    rating_data, n_users, n_items = load_rating(args)
    train_data, eval_data, test_data = dataset_split(rating_data)
    kg_pos_data = load_kg(args)
    return construct_data(args, kg_pos_data, train_data, eval_data, test_data, n_users, n_items)
    # train_data, eval_data, test_data, n_users_entities, n_relations, triplet_sets, kg_pos_data

def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final.txt'
    logging.info("load rating file: %s", rating_file)
    rating_data = pd.read_csv(rating_file, sep='\t', names=['u', 'i', 'label'])
    n_users = max(rating_data['u']) + 1
    n_items = max(rating_data['i']) + 1
    return rating_data, n_users, n_items


def dataset_split(rating_data):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_data.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    train_data = pd.DataFrame(rating_data.values[train_indices], columns=['u', 'i', 'label'])
    eval_data = pd.DataFrame(rating_data.values[eval_indices], columns=['u', 'i', 'label'])
    test_data = pd.DataFrame(rating_data.values[test_indices], columns=['u', 'i', 'label'])
    return train_data, eval_data, test_data


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final.txt'
    logging.info("locading kg file: %s", kg_file)
    kg_pos_data = pd.read_csv(kg_file, sep='\t', names=['h', 'r', 't'])
    kg_pos_data = kg_pos_data.drop_duplicates()
    return kg_pos_data


def construct_data(args, kg_pos_data, train_data, eval_data, test_data, n_users, n_items):
    # plus inverse kg data
    n_relations = max(kg_pos_data['r']) + 1
    reverse_kg_pos_data = kg_pos_data.copy()
    reverse_kg_pos_data = reverse_kg_pos_data.rename({'h': 't', 't': 'h'}, axis='columns')
    reverse_kg_pos_data['r'] += n_relations
    kg_pos_data = pd.concat([kg_pos_data, reverse_kg_pos_data], axis=0, ignore_index=True, sort=False)

    # re-map user id
    kg_pos_data['r'] += 2
    n_relations = max(kg_pos_data['r']) + 1
    n_entities = max(max(kg_pos_data['h']), max(kg_pos_data['t'])) + 1
    n_users_entities = n_users + n_entities
    print_info(n_users, n_items, n_entities, n_users_entities, n_relations)

    kg_pos_dict = collections.defaultdict(list)
    for row in kg_pos_data.iterrows():
        h, r, t = row[1]
        kg_pos_dict[h].append(t)
    kg_neg_t = []
    for row in kg_pos_data.iterrows():
        h, r, t = row[1]
        left = set(range(n_users_entities)) - set(kg_pos_dict[h])
        kg_neg_t.append(np.random.choice(list(left), 1, replace = False))

    train_data['u'] = train_data['u'] + n_entities
    eval_data['u'] = eval_data['u'] + n_entities
    test_data['u'] = test_data['u'] + n_entities

    #remove users not in train_data
    train_users = list(set(train_data['u']))
    eval_users = list(set(eval_data['u']))
    test_users = list(set(test_data['u']))

    for user in eval_users:
        if user not in train_users:
            eval_data = eval_data.drop(index=eval_data[eval_data['u']==user].index)
    eval_data = eval_data.reset_index(drop=True)

    for user in test_users:
        if user not in train_users:
            test_data = test_data.drop(index=test_data[test_data['u']==user].index)
    test_data = test_data.reset_index(drop=True)

    # add interactions to kg data
    cf2kg_train_data = pd.DataFrame(np.zeros((train_data.shape[0], 3), dtype=np.int32), columns=['h', 'r', 't'])
    cf2kg_train_data['h'] = train_data['u']
    cf2kg_train_data['t'] = train_data['i']

    reverse_cf2kg_train_data = pd.DataFrame(np.ones((train_data.shape[0], 3), dtype=np.int32), columns=['h', 'r', 't'])
    reverse_cf2kg_train_data['h'] = train_data['i']
    reverse_cf2kg_train_data['t'] = train_data['u']

    ckg_data = pd.concat([kg_pos_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)

    # construct ckg dict
    ckg_dict = collections.defaultdict(list)
    for row in ckg_data.iterrows():
        h, r, t = row[1]
        ckg_dict[h].append((t, r))
    
    triplet_sets = ckg_propagation(args, ckg_dict)
    return train_data, eval_data, test_data, n_users_entities, n_relations, triplet_sets, kg_pos_data, kg_neg_t


def ckg_propagation(args, ckg_dict):
    entity_sets = list(ckg_dict.keys())
    triplet_sets = collections.defaultdict(list)
    for obj in entity_sets:
        for layer in range(args.n_layer):
            h, r, t = [], [], []
            if layer == 0:
                entities = [obj]
            else:
                entities = triplet_sets[obj][-1][2]
            for entity in entities:
                for tail_and_relation in ckg_dict[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
            indices = np.random.choice(len(h), size=args.triplet_set_size, replace= (len(h) < args.triplet_set_size))
            h = [h[i] for i in indices]
            r = [r[i] for i in indices]
            t = [t[i] for i in indices]
            triplet_sets[obj].append((h, r, t))
    return triplet_sets


def print_info(n_users, n_items, n_entities, n_users_entities, n_relations):
    print('n_users:            %d' % n_users)
    print('n_items:            %d' % n_items)
    print('n_entities:         %d' % n_entities)
    print('n_users_entities:   %d' % n_users_entities)
    print('n_relations:        %d' % n_relations)