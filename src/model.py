import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MODEL(nn.Module):
    def __init__(self, args, n_user_entity, n_relation):
        super(MODEL, self).__init__()
        self._parse_args(args, n_user_entity, n_relation)
        self.user_entity_emb = nn.Embedding(self.n_user_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.W_R = nn.Parameter(torch.Tensor(self.n_relation, self.dim, self.dim))
        self.attention = nn.Sequential(
                nn.Linear(self.dim*3, self.dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.dim, 1),
                nn.Sigmoid(),
                ) 
        self.decay = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        self._init_weight()
    
    def calc_kg_loss(self, h, r, pos_t, neg_t):
        r_emb = self.relation_emb(r)
        W_r = self.W_R[r]

        h_emb = self.user_entity_emb(h)
        pos_t_emb = self.user_entity_emb(pos_t)
        neg_t_emb = self.user_entity_emb(neg_t.squeeze(1))

        r_mul_h = torch.bmm(h_emb.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_emb.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_emb.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_emb - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_emb - r_mul_neg_t, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        return kg_loss


    def calc_cf_score(
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,
        users_triplet: list,
        items_triplet: list,
    ):       
        user_embeddings = []
        
        # [cf_batch_size, dim]
        user_emb_0 = self.user_entity_emb(users)
        user_embeddings.append(user_emb_0)
        
        for i in range(self.n_layer):
            # [cf_batch_size, triplet_set_size, dim]
            h_emb = self.user_entity_emb(users_triplet[0][i])
            # [cf_batch_size, triplet_set_size, dim]
            r_emb = self.relation_emb(users_triplet[1][i])
            # [cf_batch_size, triplet_set_size, dim]
            t_emb = self.user_entity_emb(users_triplet[2][i])
            # [cf_batch_size, dim]
            user_emb_i = self._knowledge_attention(user_emb_0, h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)
            
        item_embeddings = []
        
        # [cf_batch_size, dim]
        item_emb_0 = self.user_entity_emb(items)
        # [cf_batch_size, dim]
        item_embeddings.append(item_emb_0)
        
        for i in range(self.n_layer):
            # [cf_batch_size, triplet_set_size, dim]
            h_emb = self.user_entity_emb(items_triplet[0][i])
            # [cf_batch_size, triplet_set_size, dim]
            r_emb = self.relation_emb(items_triplet[1][i])
            # [cf_batch_size, triplet_set_size, dim]
            t_emb = self.user_entity_emb(items_triplet[2][i])
            # [cf_batch_size, dim]
            item_emb_i = self._knowledge_attention(item_emb_0, h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)
        
        scores = self.predict(user_embeddings, item_embeddings)
        return scores
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i] * (self.decay(torch.cat((user_embeddings[0], user_embeddings[i]), dim=-1)))
                # e_u += user_embeddings[i] * 1 / (1 + i)
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i] * (self.decay(torch.cat((item_embeddings[0], item_embeddings[i]), dim=-1)))
                # e_v += item_embeddings[i] * 1 / (1 + i)
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)
            
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    

    def _parse_args(self, args, n_user_entity, n_relation):
        self.n_user_entity = n_user_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        self.l2_weight = args.l2_weight
        self.batch_size = args.cf_batch_size
        self.triplet_set_size = args.triplet_set_size


    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.user_entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    
    def _knowledge_attention(self, emb_0, h_emb, r_emb, t_emb):
        # [cf_batch_size, triplet_set_size, dim]
        init_emb = emb_0.unsqueeze(1).expand(-1,self.triplet_set_size,self.dim)
        # [cf_batch_size, triplet_set_size]
        att_weights = self.attention(torch.cat((init_emb,h_emb,r_emb),dim=-1)).squeeze(-1)
        # [cf_batch_size, triplet_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [cf_batch_size, triplet_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [cf_batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i


    def forward(self, mode, *input):
        if mode == 'calc_cf_score':
            return self.calc_cf_score(*input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)