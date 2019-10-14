#! -*- coding: utf-8 -*-
"""
@author: Gump
@time: 2019/7/11
"""
import torch
import logging
from tqdm import tqdm
import argparse

import torch.nn as nn
import torch.optim as optim

from models import EncoderInductionRelation
from utils.data_loader import DataLoader
from utils.utils import label2tensor, get_accuracy

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=5000)
parser.add_argument('--embedding_size', type=int, default=200)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--k_shot', type=int, default=10)
parser.add_argument('--q_set', type=int, default=20)
parser.add_argument('--file_path', type=str, default='./data/source.pt')
parser.add_argument('--train_epochs', type=int, default=10000)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--momentum', default=0.9)

args = parser.parse_args()

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info('{}-way, {}-shot'.format(args.n_way, args.k_shot))

# prepare data
data_loader = DataLoader()
data_loader.load(args.file_path)
embedding = data_loader.embedded(vocab_size=args.vocab_size, embedding_size=args.embedding_size)
# model
model = EncoderInductionRelation(vocab_size=args.vocab_size,
                                 embedding_size=args.embedding_size,
                                 class_num=args.n_way,
                                 hidden_size=args.hidden_size,
                                 embedding=embedding)
# optimize
loss_f = nn.MSELoss()
optimize = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# train
training_loss = 0
for epoch in range(args.train_epochs):
    batch, query_set, valid_set = data_loader.build_data(n=args.n_way, k=args.k_shot, q=args.q_set)
    train_batch = [data_loader.build_example(item) for item in batch]
    query_batch = data_loader.build_example(query_set, labeled=True)
    label_batch = [item[1] for item in query_set]

    label_example = label2tensor(label_batch)
    label_dict = label_example[0]
    logging.info('select labels:{}'.format(' '.join(list(label_dict.keys()))))

    # train
    model.train()
    class_vector, prob = model(query_set=query_batch, support_set=train_batch)

    loss = loss_f(prob, label_example[1])
    logging.info('training loss: {}'.format(loss.data))

    optimize.zero_grad()
    loss.backward()
    optimize.step()

    accuracy = get_accuracy(prob, labels=label_example[1])
    logging.info('eval accuracy: {}'.format(accuracy))

    # eval
    # model.eval()
    #
    # valid_batch = data_loader.build_example(valid_set, labeled=True)
    # valid_label = [label_dict[item[1]] for item in valid_set]
    #
    # with torch.no_grad():
    #     class_vector_valid, prob_valid = model(query_set=valid_batch, class_vector=class_vector, mode='eval')
    #     valid_accuracy = get_accuracy(prob_valid, valid_label)
    #     logging.info('eval accuracy: {}'.format(valid_accuracy))
