#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

write = SummaryWriter()


def squash(tensor):
    norm = (tensor * tensor).sum(-1)
    scale = norm / (1+norm)
    return scale.unsqueeze(-1) * tensor / torch.sqrt(norm).unsqueeze(-1)


class DynamicRouting(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicRouting, self).__init__()
        self.hidden_size = hidden_size
        self.l_1 = nn.Linear(self.hidden_size*2, self.hidden_size*2, bias=False)

    def forward(self, encoder_output, iter_routing=3):
        C, K, H = encoder_output.shape
        b = torch.zeros(C, K)
        for _ in range(iter_routing):
            d = F.softmax(b, dim=-1)
            encoder_output_hat = self.l_1(encoder_output)
            c_hat = torch.sum(encoder_output_hat*d.unsqueeze(-1), dim=1)
            c = squash(c_hat)

            b = b + torch.bmm(encoder_output_hat, c.unsqueeze(-1)).squeeze()

        # write.add_embedding(c, metadata=[0, 1, 2, 3, 4],)

        return c
