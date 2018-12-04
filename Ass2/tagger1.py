import sys
import utils as ut
import numpy as np
import torch.utils.data
from abc import abstractmethod
import matplotlib.pyplot as plt
from torch import optim, nn
import torch.nn.functional as nnf


STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

batch = 1024
default_batch = 1
embedding_dim = 50
context_size = 5
subwords_size = 3


def create_loader(fv, lv, bs):
    """
    data loader for both train and dev features and labels
    :param fv: features vectors
    :param lv: targets vectors
    :param bs: batch size
    :return: data loader
    """
    features = torch.from_numpy(np.asarray(fv)).type(torch.LongTensor)
    labels = torch.from_numpy(np.asarray(lv)).type(torch.LongTensor)
    data_set = torch.utils.data.TensorDataset(features, labels)
    data_loader = torch.utils.data.DataLoader(data_set, bs, shuffle=True)
    return data_loader


def create_test_loader(features):
    """
    data loader for test
    :param features: features of test data
    :return: data loader
    """
    return torch.from_numpy(np.asarray(features)).type(torch.LongTensor)


def init_loaders(trn_fname, d_fname, tst_fname):
    """
    initialize train/dev/test data loaders
    :param trn_fname: train file name
    :param d_fname: dev file name
    :param tst_fname: test file name
    :return: train/dev/test data loaders
    """
    # create train data loader
    features, labels = ut.create_features_and_labels_vec(trn_fname)
    train_data_loader = create_loader(features, labels, batch)
    # create dev data loader
    features, labels = ut.create_features_and_labels_vec(d_fname)
    dev_data_loader = create_loader(features, labels, default_batch)
    # create test data loader
    features = ut.create_test_features_vec(tst_fname)
    test_data_loader = create_test_loader(features)

    return train_data_loader, dev_data_loader, test_data_loader


def parse_input(args):
    """
    parse input's arguments entered by user
    :param args: list of parameters entered by user
    :return: task, tagger_type, learning_rate, epochs, hidden_dim, embedding, sub_words
    """
    task = int(args[0])
    tagger_type = args[1]
    learning_rate = float(args[2])
    epochs = int(args[3])
    hidden_dim = int(args[4])
    embedding = args[5] == '1'
    sub_words = args[6] == '1'
    return task, tagger_type, learning_rate, epochs, hidden_dim, embedding, sub_words


class NN(nn.Module):
    """
    neural network of mpl.
    work flow: one hidden layer->tanh function->softmax transformation.
    the embedding boolean sets data vectors.
    """
    def __init__(self):
        super(NN, self).__init__()
        # init embeddings without predefined weights (task 1)
        if not embedding:
            self.embeddings = nn.Embedding(len(ut.W2I), embedding_dim)
        # init embeddings with predefined weights (task 3)
        else:
            predefined_vecs = np.loadtxt("wordVectors")
            self.embeddings = nn.Embedding(predefined_vecs.shape[0], embedding_dim)
            self.embeddings.weight.data.copy_(torch.from_numpy(predefined_vecs))
        # hidden layers
        self.linear_layer1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, len(ut.T2I))

    def forward(self, inputs):
        # match the size of embeddings matrix to hidden layers
        embeds = self.embeddings(inputs).view(-1, context_size * embedding_dim)
        out_layer = torch.tanh(self.linear_layer1(embeds))
        out_layer = self.linear_layer2(out_layer)
        return nnf.log_softmax(out_layer, dim=1)


class NNF(NN):
    """
    class NNF inherit from NN
    thr logic of this net is summing prefixes and suffixes vectors with the embeddings vectors before passing
    to activation function, seems like we adding more power
    """
    def __init__(self):
        super(NNF, self).__init__()
        # init prefix and suffix embeddings
        self.prefix_emb = nn.Embedding(len(ut.P2I), embedding_dim)
        self.suffix_emb = nn.Embedding(len(ut.S2I), embedding_dim)

    def forward(self, inputs):
        # extract prefixes and suffixes from the inputs
        inputs_s = inputs.data.numpy().reshape(-1)
        prefixes = np.asanyarray([ut.P2I[ut.I2W[word_index][:subwords_size]] for word_index in inputs_s])
        suffixes = np.asanyarray([ut.S2I[ut.I2W[word_index][-subwords_size:]] for word_index in inputs_s])
        prefixes = torch.from_numpy(prefixes.reshape(inputs.data.shape)).type(torch.LongTensor)
        suffixes = torch.from_numpy(suffixes.reshape(inputs.data.shape)).type(torch.LongTensor)
        # summing up all vectors
        comp_embeddings = (self.embeddings(inputs) + self.prefix_emb(prefixes) +
                  self.suffix_emb(suffixes)).view(-1, context_size * embedding_dim)
        out_layer = torch.tanh(self.linear_layer1(comp_embeddings))
        out_layer = self.linear_layer2(out_layer)
        return nnf.log_softmax(out_layer, dim=1)


class Tagger(object):
    """
    abstract class for different taggers
    """
    def __init__(self, model):
        self.train_loader = train_dl
        self.dev_loader = dev_dl
        self.test_loader = test_dl
        self.nn_model = model
        self.dev_accuracy_for_epoch = {}
        self.dev_loss_for_epoch = {}
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=l_rate)

    def train(self):
        """
        trains the model.
        for each epoch keep accuracy and loss in dev loader
        :return:
        """
        print('model is being trained..')
        self.nn_model.train()
        # run throw all epochs
        for epoch in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.nn_model(data)
                loss = nnf.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
            dev_accuracy, dev_loss = self.accuracy_and_loss()
            self.dev_accuracy_for_epoch[epoch] = dev_accuracy
            self.dev_loss_for_epoch[epoch] = dev_loss


    def predict(self):
        """
        predict labels of test loader
        :return: list of predicted tags
        """
        print('in prediction process..')
        prediciotns = []
        self.nn_model.eval()
        for data in self.test_loader:
            output = self.nn_model(data)
            pred = output.data.max(1)[1]
            prediciotns.append(ut.I2T[pred.item()])
        return prediciotns


    def create_graphs(self):
        """
        create the following graphs:
            1. dev accuracy for number of epochs
            2. dev loss for number of epochs
        :return:
        """
        print('plot two graphs for accuracy and loss')
        acc_epochs = self.dev_accuracy_for_epoch.keys()
        accuracies = self.dev_accuracy_for_epoch.values()
        dev_epochs = self.dev_loss_for_epoch.keys()
        losses = self.dev_loss_for_epoch.values()
        # create first graph
        plt.plot(acc_epochs, accuracies, 'b')
        plt.title('Accuracy graph')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(accuracy_file_name)
        plt.clf()
        # create second graph
        plt.plot(dev_epochs, losses , 'r')
        plt.title('Loss graph')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(loss_file_name)


    @staticmethod
    def output_pred(pred_list):
        """
        predict tags of words from test file.
        keep all results in predictions list.
        :param pred_list: list of tags
        :return:
        """
        output = []
        e_lines = 0
        test_file = open(test_file_name, 'r')
        for index, line in enumerate(test_file):
            if not line.strip():
                e_lines += 1
                output.append('')
                continue
            word = line.strip()
            output.append(word + ' ' + pred_list[index - e_lines])
        pred_file = open(pred_file_name, 'w')
        pred_file.write('\n'.join(output))
        print('tags of test file were predicted.')

    @abstractmethod
    def accuracy_and_loss(self):
        """
        abstract function, children should implement
        :return:
        """
        pass


class PosTagger(Tagger):
    """
    PostTagger inherit from Tagger.
    implementing single func of which calculates accuracy and loos
    """
    def accuracy_and_loss(self):
        """
        calc accuracy and loss for dev loader
        :return: accuracy and loss
        """
        loss = correct = 0
        for data, target in self.dev_loader:
            output = self.nn_model(data)
            loss += nnf.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()
        loss = loss / len(self.dev_loader.dataset)
        accuracy = (100. * correct.item()) / len(self.dev_loader.dataset)
        print('Loss: ' + '%.3f' % loss + '\n' + 'Accuracy: ' + '%.3f' % accuracy + '%')
        return accuracy, loss



class NerTagger(Tagger):
    """
    NerTagger inherit from Tagger.
    implementing single func of which calculates accuracy and loss
    """
    def accuracy_and_loss(self):
        """
        calc accuracy and loss for dev loader
        :return: accuracy and loss
        """
        loss = correct = total = 0
        for data, target in self.dev_loader:
            net_out = self.nn_model(data)
            loss += nnf.nll_loss(net_out, target).item()
            pred = net_out.data.max(1)[1]
            if ut.I2T[pred.item()] != 'O' or ut.I2T[target.item()] != 'O':
                correct += pred.eq(target.data).sum()
                total += 1
        loss = loss / len(self.dev_loader.dataset)
        accuracy = (100. * correct.item()) / total
        print('Loss: ' + '%.3f' % loss + '\n' + 'Accuracy: ' + '%.3f' % accuracy + '%')
        return accuracy, loss

def init_model_and_tagger():
    model = NN()
    if sub_words:
        ut.create_additional_dict(subwords_size)
        model = NNF()
    else:
        model = NN()
    if tag_type == 'pos':
        tagger = PosTagger(model)
    else:
        tagger = NerTagger(model)
    # train model
    tagger.train()
    # plot
    tagger.create_graphs()
    # predict
    predict_list = tagger.predict()
    tagger.output_pred(predict_list)


if __name__ == '__main__':
    [task, tag_type, l_rate, epochs, hidden_dim, embedding, sub_words] = parse_input(sys.argv[1:])
    train_file_name = tag_type + '/train'
    dev_file_name = tag_type + '/dev'
    test_file_name = tag_type + '/test'
    pred_file_name =  tag_type + '/test' + str(task) + '.' + tag_type
    accuracy_file_name = tag_type + '/accuracy_plot' + str(task) + '.' + 'png'
    loss_file_name = tag_type + '/loss_plot' + str(task) + '.' + 'png'
    # process data
    ut.extract_words_and_tags(train_file_name)
    ut.create_dicts()
    # init data loaders
    train_dl, dev_dl, test_dl = init_loaders(train_file_name, dev_file_name, test_file_name)
    # init model and tagger
    init_model_and_tagger()

