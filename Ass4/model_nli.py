import torch
import torch.nn as nn
import numpy as np


class NLINet(nn.Module):
    """
    Natural Language Inference network
    """
    def __init__(self, config):
        super(NLINet, self).__init__()

        # Bi directional LSTM with max pooling encoder
        self.encoder = BiLSTMMaxPoolEncoder(config)
        self.bilstm_dim = config['bilstm_dim']
        self.input_dim = 4 * 2 * self.bilstm_dim

        # classifier
        self.dpout_fc = config['dropout']
        self.mlp_dim = config['mlp_dim']
        self.output_dim = config['output_dim']
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dpout_fc),
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dpout_fc),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dpout_fc),
            nn.Linear(self.mlp_dim, self.output_dim)
        )

    def forward(self, prem_stentences, hypo_sentences):
        u = self.encoder(prem_stentences)
        v = self.encoder(hypo_sentences)

        features = torch.cat([u, v, torch.abs(u-v), u*v], 1)
        output = self.classifier(features)
        return output



class BiLSTMMaxPoolEncoder(nn.Module):
    """
    Bi-directional LSTM with max pooling encoder
    """
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.bilstm_dim = config['bilstm_dim']
        self.lstm_layers = config['lstm_layers']
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.bilstm_dim, self.lstm_layers,
                                bidirectional=True)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, index_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(index_sort)

        index_sort = torch.from_numpy(index_sort)
        sent = sent.index_select(1, index_sort)

        # Handling padding in Recurrent Network
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*lstm_dim
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # pooling
        embeddings = torch.max(sent_output, 0)[0]
        return embeddings
