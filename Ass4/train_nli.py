import argparse
import utils as ut
import extract_data as ed
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model_nli import NLINet
import os


"""
ARGUMENTS FOR PARSER
"""
parser = argparse.ArgumentParser(description='NLI training')

# paths
parser.add_argument("--nli_path", type=str, default='data/snli/', help="NLI data path")
parser.add_argument("--output_dir", type=str, default='.output', help="output directory")
parser.add_argument("-model_name", type=str, default='model.pickle')
parser.add_argument("--vocabulary_path", type=str, default='data/vocabulary/vocabulary.pickle',
                    help="vocabulary file path")

# training
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")

# model
parser.add_argument("--bilstm_dim", type=int, default=4096, help="bilstm hidden dimension")
parser.add_argument("--lstm_layers", type=int, default=1, help="lstm number of layers")
parser.add_argument("--mlp_dim", type=int, default=512, help="hidden dimension of mlp layers")
parser.add_argument("--dropout", type=float, default=0., help="dropout")
parser.add_argument("--output_dim", type=int, default=3, help="entailment/neutral/contradiction")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embeddings dimension")


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
# loss
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(nli_net.parameters(), config.learning_rate)

"""
TRAIN
"""
dev_acc_best = -1e10
stop_training = False


def train_epoch(ep):
    print('training: epoch {0}'.format(ep))
    nli_net.train()
    correct = 0.0
    premises = train['premise']
    hypothesises = train['hypothesis']
    targets = train['label']

    if ep > 1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.decay

    for i in range(0, len(premises), config.batch_size):
        prem_batch, prem_len = ut.get_batch(premises[i:i + config.batch_size],
                                                    word_vectors, config.word_emb_dim)
        hypo_batch, hypothesises_len = ut.get_batch(hypothesises[i:i + config.batch_size],
                                                            word_vectors, config.word_emb_dim)
        prem_batch, hypo_batch = Variable(prem_batch), Variable(hypo_batch)
        targets_batch = Variable(torch.LongTensor(targets[i:i + config.batch_size]))
        # forward
        output = nli_net((prem_batch, prem_len), (hypo_batch, hypothesises_len))
        # predictions
        pred = output.data.max(1)[1]
        correct += pred.long().eq(targets_batch.data.long()).cpu().sum().item()
        # calculate loss
        loss = loss_fn(output, targets_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # optimizer step
        optimizer.step()

    # calculate overall accuracy on train
    train_acc = round(100 * correct / len(premises), 2)
    print('results : epoch {0} , accuracy: {1}'.format(ep, train_acc))


def evaluate(ep):
    print('validation : Epoch {0}'.format(ep))
    nli_net.eval()
    correct = 0.
    global dev_acc_best, stop_training
    premises = dev['premise']; hypothesises = dev['hypothesis']; targets = dev['label']

    for i in range(0, len(premises), config.batch_size):
        # prepare batch
        prem_batch, premises_len = ut.get_batch(premises[i:i + config.batch_size],
                                                    word_vectors, config.word_emb_dim)
        hypo_batch, hypo_len = ut.get_batch(hypothesises[i:i + config.batch_size],
                                                            word_vectors, config.word_emb_dim)
        prem_batch, hypo_batch = Variable(prem_batch), Variable(hypo_batch)
        targets_batch = Variable(torch.LongTensor(targets[i:i + config.batch_size]))
        # model forward
        output = nli_net((prem_batch, premises_len), (hypo_batch, hypo_len))
        # predictions
        pred = output.data.max(1)[1]
        correct += pred.long().eq(targets_batch.data.long()).cpu().sum().item()

    # calculate overall accuracy on dev
    eval_acc = round(100 * correct / len(premises), 2)
    print('results: epoch {0} , accuracy by dev {1}'.format(ep, eval_acc))

    if eval_acc > dev_acc_best:
        # save best model so far
        print('saving model at epoch {0}'.format(epoch))
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        torch.save(nli_net.state_dict(), os.path.join(config.output_dir, config.model_name))
        dev_acc_best = eval_acc
    else:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / config.lrshrink
        if optimizer.param_groups[0]['lr'] < config.minlr:
            stop_training = True


"""
Train model on Natural Language Inference task
"""
start = time.time()
epoch = 1

while not stop_training and epoch <= config.epochs:
    train_epoch(epoch)
    evaluate(epoch)
    epoch += 1

end = time.time()
print('time the model was trained: {}'.format(str(end - start)))
