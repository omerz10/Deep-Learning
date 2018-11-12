import loglinear as ll
import random
import utils as ut
import numpy as np

STUDENT={'name': 'Omer Zucker_Omer Wolf',
         'ID': '200876548_307965988'}


def feats_to_vec(features):
    """
    convert features into vector
    :param features
    :return: binary vector
    """
    vec = np.zeros(len(ut.F2I))
    for feature in features:
        if feature in ut.F2I:
            vec[ut.F2I[feature]] += 1
    return vec

def accuracy_on_dataset(dataset, params):
    """
    calculates the accuracy of the prediction on a given data set
    :param dataset: bigrams of 2 letters and languages
    :param params, dataset
    :return: accuracy of the prediction by precentage
    """
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = ut.L2I[label]
        # prediction returned the correct label
        if ll.predict(x, params) == y:
            good +=1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)         # convert features to a vector.
            y = ut.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            params = np.subtract(params, np.multiply(learning_rate, grads))
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def create_test_file(data_set, params):
    """
    create file with results of languages
    :param data_set: bigrams of 2 letters
    :param params
    :return: file with result
    """
    test_file = open("test.pred",'w')
    for l, features in data_set:
        x = feats_to_vec(features)
        index = ll.predict(x, params)
        for key, value in ut.L2I.items():
            if value == index:
                l = key
                break
        test_file.write(l+"\n")
    test_file.close()

if __name__ == '__main__':

    in_dim = len(ut.F2I)
    out_dim = len(ut.L2I)
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(ut.TRAIN, ut.DEV, 10, 0.01, params)

    TEST = [(l, ut.text_to_bigrams(t)) for l, t in ut.read_data("test")]
    create_test_file(TEST, trained_params)
