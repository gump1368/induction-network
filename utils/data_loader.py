#! -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import collections

import torch

from utils import is_ch, get_word_vector


PAD = ''
LABELS = ['alarm', 'bag', 'chat', 'command', 'face', 'food', 'game', 'greet', 'intelligent_home', 'machine', 'music',
          'news', 'query', 'radio', 'sleep', 'story', 'study', 'time', 'volume', 'weather']


class DataLoader(object):
    def __init__(self):
        self.labels = LABELS
        self.words_counts = collections.defaultdict(int)
        self.data = collections.defaultdict(list)
        self.words2id = collections.defaultdict(int)
        self.id2words = collections.defaultdict(str)
        self.word_embedding = collections.defaultdict(list)

    def read_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().replace('__label__', '').split()
                label = line[0]
                words = line[1:]

                self.data[label].append(words)
                for word in words:
                    self.words_counts[word] += 1

    def build_vocab(self, vocab_path, vocab_size=5000):
        sorted_words = sorted(self.words_counts.items(), key=lambda x: x[1], reverse=True)
        filter_words = list(filter(lambda x: is_ch(str(x[0])), sorted_words))

        # save vocabulary
        write = open(vocab_path, 'w', encoding='utf-8')

        self.words2id['UNK'] = 0
        self.id2words[0] = 'UNK'
        index = 1

        for item in filter_words:
            word = item[0]
            write.write(word+'\n')

            word_embedding = get_word_vector(word)
            if not word_embedding:
                continue

            self.words2id[word] = index
            self.id2words[index] = word
            self.word_embedding[word] = word_embedding
            index += 1

            if index == vocab_size:
                break

    def build_data(self, n, k, q):
        batch = []
        query_set = []
        query_label = []
        valid = []
        valid_label = []
        c_way = random.sample(self.labels, k=n)
        for c in c_way:
            batch_sample = random.sample(self.data[c], k=k)
            batch.append(sorted(batch_sample, key=lambda x: len(x), reverse=True))

            query_set.extend(random.sample(self.data[c], k=q))
            query_label.extend([c]*q)

            valid.extend(random.sample(self.data[c], k=10))
            valid_label.extend([c]*10)

        query_set = sorted(zip(query_set, query_label), key=lambda x: len(x[0]), reverse=True)
        valid_set = sorted(zip(valid, valid_label), key=lambda x: len(x[0]), reverse=True)
        return batch, query_set, valid_set

    def build_example(self, batch, max_length=20, labeled=False):
        seq_tokens = []
        seq_length = []
        mask_total = []
        for seq in batch:
            if labeled:
                seq = seq[0]

            if len(seq) > max_length:
                seq = seq[:max_length]
                # print('Too long sequence! Please build batch again!')
                # return None

            length = len(seq)
            seq_token = [self.words2id[word] for word in seq]
            padding = [0] * (max_length-length)
            seq_token = seq_token + padding
            mask = [1]*length+padding

            seq_tokens.append(seq_token)
            seq_length.append(length)
            mask_total.append(mask)

        example = (torch.LongTensor(seq_tokens), torch.LongTensor(seq_length), torch.LongTensor(mask_total))
        return example

    def embedded(self, vocab_size, embedding_size):
        embedding = torch.FloatTensor(vocab_size, embedding_size)
        for _id, word in self.id2words.items():
            if word in self.word_embedding:
                embedding[_id] = torch.FloatTensor(self.word_embedding[word])
            else:
                embedding[_id] = torch.zeros(embedding_size)
        return embedding

    def save(self, path):
        save_data = {
            'data': self.data,
            'word2id': self.words2id,
            'id2word': self.id2words,
            'embedding': self.word_embedding
        }

        torch.save(save_data, path)

    def load(self, path):
        data = torch.load(path)
        self.data = data['data']
        self.word_embedding = data['embedding']
        self.words2id = data['word2id']
        self.id2words = data['id2word']


if __name__ == '__main__':
    source = '/home/gump/Software/pycharm-2018.1.6/projects/text-classification/induction-network/'
    data_path = os.path.join(source, 'data/data_3.txt')
    data_loader = DataLoader()
    data_loader.read_data(data_path)
    data_loader.build_vocab(os.path.join(source, 'data/vocabulary.txt'))
    data_loader.save(os.path.join(source, 'data/source.pt'))





