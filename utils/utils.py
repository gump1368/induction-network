#! -*- coding: utf-8 -*-
import torch
import numpy as np
import requests
import collections


def is_ch(string):
    for char in string:
        if not 0x4e00 <= ord(char) <= 0x9fa6:
            return False
    return True


def get_word_vector(word):
    url = ' http://172.27.1.207:11109/vector'
    data = {
        "type": "word_vector",
        "data": {
            "word_list": [word]
        }
    }
    try:
        resp = requests.post(url, json=data).json()
    except:
        print('cannot get vector of word: {}'.format(word))
        return []
    return resp['data'][word]


def get_accuracy(prob, labels):
    index = np.array(torch.argmax(prob, dim=1)).tolist()
    labels = np.array(torch.argmax(labels, dim=1)).tolist()
    print(index)
    length = len(labels)
    num = 0
    for i in range(length):
        if index[i] == labels[i]:
            num += 1

    return num/length


def label2tensor(label):
    label_dict = {}
    for index, _label in enumerate(set(label)):
        label_dict[_label] = index

    label2id = torch.LongTensor([[label_dict[k]] for k in label])
    label_one_hot = torch.zeros(len(label), len(label_dict)).scatter_(1, label2id, 1)

    return label_dict, label_one_hot
