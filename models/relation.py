#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter


def k_max_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class Relation(nn.Module):
    def __init__(self, hidden_size, class_num, output_size=100, k_max=3):
        super(Relation, self).__init__()
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k_max = k_max

        self.M = nn.init.xavier_normal_(torch.FloatTensor(hidden_size * 2, hidden_size * 2, self.output_size))
        self.M.requires_grad = True

        self.l1 = nn.Linear(self.hidden_size*2, 1)
        self.l2 = nn.Linear(self.output_size, 1)

        self.fc = nn.Linear(self.hidden_size*2*self.output_size, class_num, bias=True)

    def forward(self, class_vector, query_vector):
        query_size, hidden_size = query_vector.shape
        # query_vector = query_vector.unsqueeze(1).repeat(1, self.class_num, 1)
        # class_vector = class_vector.unsqueeze(0).repeat(query_size, 1, 1)

        # concatenate
        # concatenated = torch.cat([query_vector, class_vector], dim=-1)

        # concatenated = self.l1(concatenated).squeeze()
        # concatenated = concatenated.view(self.output_size, query_size, self.class_num)
        mid_pro = []
        for i in range(self.output_size):
            v = self.M[:, :, i]
            inter = torch.mm(query_vector, torch.mm(class_vector, v).transpose(0, 1))

            # inter = torch.sum(torch.mul(torch.matmul(class_vector.unsqueeze(-1), query_vector.unsqueeze(2)), v), dim=2)
            # inter= torch.sum(inter*v, dim=1)
            # inter = (self.M[:, :, i].unsqueeze(0).repeat(query_size, 1, 1))*torch.bmm(query_vector, class_vector.transpose(1, 2))
            mid_pro.append(inter)
        tensor_bi_product = torch.stack(mid_pro, dim=0)  # (C*K,Q)
        activate = F.relu(tensor_bi_product)
        reshape = activate.permute(1, 2, 0)
        other = self.l2(reshape).squeeze()
        # other = self.l1(other).view(query_size, self.class_num)
        # tensor_bi_product = concatenated+tensor_bi_product

        # k_max = k_max_pooling(F.tanh(tensor_bi_product), dim=2, k=self.k_max).view(query_size, self.k_max*self.output_size)
        # output = self.fc(F.relu(tensor_bi_product))
        probability = torch.sigmoid(other)
        return probability
