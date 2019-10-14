#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.encoder import Encoder
from models.induction import DynamicRouting
from models.relation import Relation


class EncoderInductionRelation(nn.Module):
    def __init__(self, vocab_size, embedding_size, class_num, hidden_size, embedding=None):
        super(EncoderInductionRelation, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.encoder = Encoder(vocab_size=self.vocab_size,
                               hidden_size=self.hidden_size,
                               embedding_size=self.embedding_size,
                               embedding=self.embedding)

        self.dynamic_routing = DynamicRouting(hidden_size=self.hidden_size)

        self.relation = Relation(hidden_size=self.hidden_size, class_num=self.class_num)

    def forward(self, query_set, support_set=None, class_vector=None, mode='train'):
        query_encoder = self.encoder(query_set)

        if mode == 'train':
            support_encoder = torch.stack([self.encoder(item) for item in support_set])
            class_vector = self.dynamic_routing(support_encoder)  # (C, H)

        score = self.relation(class_vector, query_encoder)

        return class_vector, score
