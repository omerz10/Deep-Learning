import argparse
import utils as ut
import extract_data as ed
import numpy as np
import torch
from torch.autograd import Variable
from model_nli import NLINet
import os


"""
ARGUMENTS FOR PARSER
"""
parser = argparse.ArgumentParser(description='NLI testing')

# paths
parser.add_argument("--nli_path", type=str, default='data/snli/', help="NLI data path")
parser.add_argument("--output_dir", type=str, default='.output', help="output directory")
parser.add_argument("--model_name", type=str, default='model.pickle')
parser.add_argument("--vocabulary_path", type=str, default='data/vocabulary/vocabulary.pickle',
                    help="vocabulary file path")
# testing
parser.add_argument("--batch_size", type=int, default=64)

# model
parser.add_argument("--bilstm_dim", type=int, default=4096, help="bilstm hidden dimension")
parser.add_argument("--lstm_layers", type=int, default=1, help="lstm num layers")
parser.add_argument("--dropout", type=float, default=0., help="classifier dropout")
parser.add_argument("--mlp_dim", type=int, default=512, help="hidden dim of mlp layers")
parser.add_argument("--output_dim", type=int, default=3, help="entailment/neutral/contradiction")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")


"""
CONFIGURATIONS
"""
config = parser.parse_args()

"""
DATA
"""
train, dev, test = ed.import_datasets(config.nli_path)
word_vectors = ut.load_vocabulary(config.vocabulary_path)

for sentence_type in ['premise', 'hypothesis']:
    for data_type in ['train', 'dev']:
        eval(data_type)[sentence_type] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vectors] +
                                           ['</s>'] for sent in eval(data_type)[sentence_type]])
"""
MODEL
"""
# model configurations
config_nli_model = {
    'word_emb_dim': config.word_emb_dim,
    'bilstm_dim': config.bilstm_dim,
    'lstm_layers': config.lstm_layers,
    'dropout': config.dropout,
    'mlp_dim': config.mlp_dim,
    'output_dim': config.output_dim
}

# model
nli_net = NLINet(config_nli_model)

"""
TEST
"""
def test_m():
    print('testing..')
    nli_net.eval()
    correct = 0.0
    premises = test['premise']; hypothesises = test['hypothesis']; targets = test['label']

    for i in range(0, len(premises), config.batch_size):
        prem_batch, prem_len = ut.get_batch(premises[i:i + config.batch_size],
                                                    word_vectors, config.word_emb_dim)
        hypoth_batch, hypothesises_len = ut.get_batch(hypothesises[i:i + config.batch_size],
                                                            word_vectors, config.word_emb_dim)
        prem_batch, hypoth_batch = Variable(prem_batch), Variable(hypoth_batch)
        targets_batch = Variable(torch.LongTensor(targets[i:i + config.batch_size]))
        # model forward
        output = nli_net((prem_batch, prem_len), (hypoth_batch, hypothesises_len))
        # predictions
        pred = output.data.max(1)[1]
        correct += pred.long().eq(targets_batch.data.long()).cpu().sum().item()
    # calculate accuracy of test
    test_acc = round(100 * correct / len(premises), 2)
    print('accuracy on test {0}'.format(test_acc))


"""
Test model on Natural Language Inference task
"""
nli_net.load_state_dict(torch.load(os.path.join(config.output_dir, config.model_name)))
test_m()
