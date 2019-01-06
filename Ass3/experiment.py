import numpy as np
import dynet as dy
import sys
import random
import time

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

VOCABULARY = set()
TAGS = set()
C2I = dict()
I2C = dict()
T2I = dict()
I2T = dict()
EMBEDDINGS_DIM = 180
LAYERS = 1
HIDDEN_DIM = 50
HIDDEN_MLP_DIM = 50
EPOCHS = 1
BATCH = 360  # number of sentences


def parse_from_file(filename):
    """
    parsing data from a given file name.
    for each line create a tuple of {word : tag}
    :param filename: file name
    :return: list of tuples
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            word, tag = line.strip().split()
            data.append((word, tag))
    return data


def create_dictionaries(train_set):
    """
    create dictionaries for converting characters into indexes and vice versa.
    Same for tags and their indexes.
    :param train_set: train data set
    """
    global C2I, I2C, T2I, I2T
    # run through all lines in data set
    for word, tag in train_set:
        for c in word:
            VOCABULARY.add(c)
        TAGS.add(tag)
    C2I = {c: i for i, c in enumerate(VOCABULARY)}
    I2C = {i: c for c, i in C2I.items()}
    T2I = {c: i for i, c in enumerate(TAGS)}
    I2T = {i: c for c, i in T2I.items()}


class LSTMacceptor(object):
    """
    initialize LSTM acceptor model of 1 layer with MLP of one hidden layer
    """
    def __init__(self):
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.embeddings = self.model.add_lookup_parameters((len(C2I), EMBEDDINGS_DIM))
        self.builder = dy.LSTMBuilder(LAYERS, EMBEDDINGS_DIM, HIDDEN_DIM, self.model)
        # MLP - hidden layer
        self.W1 = self.model.add_parameters((HIDDEN_MLP_DIM, HIDDEN_DIM))
        self.b1 = self.model.add_parameters(HIDDEN_MLP_DIM)
        # MLP - output layer
        self.W2 = self.model.add_parameters((len(T2I), HIDDEN_MLP_DIM))
        self.b2 = self.model.add_parameters(len(T2I))


    def __call__(self, word):
        """
        initialize new computation graph by previous parameters.
        run prediction of a given word.
        :param word: word for prediction
        :return: prediction result
        """
        dy.renew_cg()
        embeddings = self.represent(word)
        init_state = self.builder.initial_state()
        output = init_state.transduce(embeddings)[-1]
        result = self.W2 * (dy.tanh(self.W1 * output + self.b1)) + self.b2
        return dy.softmax(result)


    def represent(self, word):
        """
        set embedding of character representation of a given word
        :param word: word for embedding
        :return: embeddings
        """
        characters_index = [C2I[character] for character in word]
        embeddings = [self.embeddings[i] for i in characters_index]
        return embeddings


    def compute_loss(self, word, tag):
        """
        return negative cross entropy loss of a given word and its tag
        :param word: word for prediction
        :param tag: gold tag
        :return: loss
        """
        word_t = self(word)
        loss = -dy.log(dy.pick(word_t, T2I[tag]))
        return loss


    def compute_prediction(self, word):
        """
        predict the label of a given word
        :param word: word for prediction
        :return: predicted label
        """
        word_t = self(word)
        return I2T[np.argmax(word_t.value())]


def compute_accuracy(dev_set, model):
    """
    compute the accuracy of the model
    :param dev_set: dev data set
    :param model: LSTM acceptor
    :return: computed accuracy
    """
    correct = 0.0
    for word, tag in dev_set:
        label = model.compute_prediction(word)
        if label == tag:
            correct += 1
    return 100 * (correct / len(dev_set))


def train(train_set, dev_set, model):
    """
    train LSTM acceptor model using train set.
    print average loss for each epoch.
    print average loss and accuracy of dev set.
    :param train_set: train data set
    :param dev_set: dev data set
    :param model: LSTM acceptor
    """
    init_time = time.time()
    for epoch in range(EPOCHS):
        loss_sum = 0.0
        random.shuffle(train_set)
        for index, (word, tag) in enumerate(train_set, 1):
            if index % BATCH == 0:
                single_accuracy = compute_accuracy(dev_set, model)
                print('Accuracy' + ': ' + str(single_accuracy) + '%')
            loss = model.compute_loss(word, tag)
            loss_sum += loss.value()
            loss.backward()
            model.trainer.update()
        print('Epochs: ' + str(epoch+1) + '\nAverage loss: ' + str(loss_sum / len(train_set)))
    end_time = time.time()
    print('Time the model being trained: ' + str(end_time - init_time))


if __name__ == '__main__':
    # parse inout
    train_file_path, dev_file_path = sys.argv[1:]

    # data creation
    train_set = parse_from_file(train_file_path)
    create_dictionaries(train_set)
    dev_set = parse_from_file(dev_file_path)

    # model creation
    model = LSTMacceptor()

    # train the model
    print('model is being trained..')
    train(train_set, dev_set, model)
