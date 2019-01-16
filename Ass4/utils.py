import numpy as np
import pickle
import torch


def load_vocabulary(path):
    print('loading vocabulary..')
    with open(path, 'rb') as handle:
        vocabulary = pickle.load(handle)
    print('vocabulary was loaded successfully!')
    return vocabulary


def get_batch(batch, word_vectors, word_emb_dim=300):
    # sent in batch in decreasing order of lengths
    # batch: (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embeddings = np.zeros((max_len, len(batch), word_emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embeddings[j, i, :] = word_vectors[batch[i][j]]
    return torch.from_numpy(embeddings).float(), lengths
