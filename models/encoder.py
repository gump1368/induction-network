#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.attention import SelfAttention


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, bidirectional=True, embedding=None):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.gru = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=self.bidirectional, batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        self.attention = SelfAttention(hidden_size=self.hidden_size)

    def forward(self, batch):
        tokens, lengths, mask = batch

        input_seq = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(input_seq, lengths, batch_first=True)

        outputs, hidden = self.gru(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # if self.bidirectional:
        #     outputs = outputs[:, :, self.hidden_size:] + outputs[:, :, :self.hidden_size]

        return self.attention(outputs, mask)
