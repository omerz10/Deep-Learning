import os
import sys
import utils as ut
import random
import time
import models as bm
import pickle
from zipfile import ZipFile

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

EPOCHS = 50
DATA_RATIO = 0.9
BATCH = 500
TRAIN_SIZE = 15000


def save_model(model, model_file_path):
    """
    set model related data into a file
    :param model: given model
    :param model_file_path: model file path
    """
    # set dictionaries to dump
    with open('data.pkl', 'wb') as file:
        pickle.dump(ut.W2I, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.T2I, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.I2T, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.C2I, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.P2I, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.S2I, file, pickle.HIGHEST_PROTOCOL)
    # set all data to a file
    model.model.save('model')


def compute_accuracy(data, model, data_type):
    """
    compute accuracy of the model
    :param data_type: pos/ner
    :param data: dev data set
    :param model: BiLSTM model
    :return:
    """
    bad = good = 0.0
    for sentence, tags in data:
        # computing model's predictions on the sentence
        s_pred_tags = model.compute_prediction(sentence)
        for pred_tag, v_tag in zip(s_pred_tags, tags):
            if data_type == 'ner' and pred_tag == 'O' and v_tag == 'O':
                pass
            elif pred_tag == v_tag:
                good += 1
            else:
                bad += 1
    return 100 * (good / (good + bad))


def init_model(repr):
    """
    initialize by given representation
    :param repr: representation of word
    :return: a model
    """
    if repr == 'a':
        return bm.Amodel(ut.W2I, ut.T2I, ut.I2T)
    elif repr == 'b':
        return bm.Bmodel(ut.W2I, ut.T2I, ut.I2T, ut.C2I)
    elif repr == 'c':
        return bm.Cmodel(ut.W2I, ut.T2I, ut.I2T, ut.P2I, ut.S2I)
    else:   # repr == 'd''
        return bm.Dmodel(ut.W2I, ut.T2I, ut.I2T, ut.C2I)


def train(train_data, dev_data, model, data_type):
    """
    train the model
    :param train_data: train data set
    :param dev_data: dev data set
    :param model: BiLSTM model
    :param data_type: ner/pos
    :return: accuracy graph
    """
    graph = {}
    init_time = time.time()
    for epoch in range(EPOCHS):
        loss_sum = tagged_words = 0.0
        random.shuffle(train_data)
        for index, (sentence, tags) in enumerate(train_data, 1):
            if index % BATCH == 0:
                accuracy = compute_accuracy(dev_data, model, data_type)
                print('Dev accuracy: ' + str(accuracy) + '%')
                graph[index / 100] = accuracy
            loss = model.compute_loss(sentence, tags)
            loss_sum += loss.value()
            tagged_words += len(tags)
            loss.backward()
            model.trainer.update()
        print('Train data set results:\n\tEpoch: '+str(epoch + 1)+'\n\t'+'Average Loss: '+str(loss_sum/tagged_words))
    end_time = time.time()
    print('\nTime the model being trained: ' + str(end_time - init_time))
    return graph


if __name__ == '__main__':

    # parse input
    repr, train_filename, model_filename, data_type = sys.argv[1:]
    train_file_path = str(data_type) + '/' + str(train_filename)
    model_file_path = str(data_type) + '/' + str(repr) + '_' + str(model_filename)

    # data creation
    data = ut.parse_data_from_file(train_file_path)[:TRAIN_SIZE]
    ut.create_dictionaries(data)
    train_data = data[:(int(len(data) * DATA_RATIO))]
    dev_data = data[-(int(len(data) * (1 - DATA_RATIO))):]

    # initialize model
    model = init_model(repr)

    # train
    print('model is being trained..')
    accuracy_graph = train(train_data, dev_data, model, data_type)

    # create accuracy graph ans save model
    graph_file_path = str(data_type) + '/' + str(repr) + '_' + 'graph.pkl'
    with open(graph_file_path, 'wb') as graph_file:
        pickle.dump(accuracy_graph, graph_file, pickle.HIGHEST_PROTOCOL)
    save_model(model, model_file_path)

