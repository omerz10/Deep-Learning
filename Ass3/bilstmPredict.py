import sys
import utils as ut
import os
import models
import time
from zipfile import ZipFile
import pickle

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}


def predict(data, model):
    """
    predict by a given model and data
    :param data: unresolved data before prediction
    :param model: trained model for predictions
    :return: result of prediction
    """
    result = []
    init_time = time.time()
    for sentence in data:
        result.append(model.compute_prediction(sentence))
    end_time = time.time()
    print('\nTime for prediction: ' + str(end_time - init_time))
    return result


def init_model(repr, model_file_path):
    """
    initialize model
    :param repr: format of representation
    :param model_file_path: a file path
    :return: a model
    """
    # load dictionaries from file
    with open('data.pkl', 'rb') as dicts_file:
        w2i = pickle.load(dicts_file)
        t2i = pickle.load(dicts_file)
        i2t = pickle.load(dicts_file)
        c2i = pickle.load(dicts_file)
        p2i = pickle.load(dicts_file)
        s2i = pickle.load(dicts_file)
    # select model by representation
    if repr == 'a':
        model =  models.Amodel(w2i, t2i, i2t)
    elif repr == 'b':
        model = models.Bmodel(w2i, t2i, i2t, c2i)
    elif repr == 'c':
        model = models.Cmodel(w2i, t2i, i2t, p2i, s2i)
    else:
        model = models.Dmodel(w2i, t2i, i2t, c2i)
    model.model.populate('model')
    os.remove("data.pkl")
    os.remove("model")
    return model


if __name__ == '__main__':

    # parse input
    repr, model_file_name, input_filename, data_type = sys.argv[1:]
    input_file_path = str(data_type) + '/' + str(input_filename)
    model_file_path = str(data_type) + '/' + str(repr) + '_' + str(model_file_name)
    results_file_path = str(data_type) + '/' + str(repr) + '_' + 'results'

    # data creation
    unresolved_data = ut.create_unresolved_data(input_file_path)

    # initialize model
    model = init_model(repr, model_file_path)

    # predict and set results
    print('model is predicting..')
    pred_data = predict(unresolved_data, model)
    ut.create_results_file(pred_data, unresolved_data, results_file_path)
