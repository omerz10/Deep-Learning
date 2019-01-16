import numpy as np
import os
import json
import pickle
import nltk


# paths
SNLI_PATH = 'data/snli/'
GLOVE_PATH = 'data/glove/glove.840B.300d.txt'
VOCABULARY_PATH = 'data/vocabulary/vocabulary.pickle'


def import_datasets(snli_path):
    """
    import datasets from json file within data/snli directory
    :param snli_path: path
    :return: train_data, dev_data, test_data
    """
    print('extract data from snli directory..')
    train = dict(); dev = dict(); test = dict()
    gold_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    for file_type in ['train', 'dev', 'test']:
        path = os.path.join(snli_path, 'snli_1.0_{}.jsonl'.format(file_type))
        with open(path) as file:
            data = [json.loads(line) for line in file]
        eval(file_type)['premise'] = [entry['sentence1'] for entry in data if entry['gold_label'] != '-']
        eval(file_type)['hypothesis'] = [entry['sentence2'] for entry in data if entry['gold_label'] != '-']
        g_labels = np.array([gold_labels[entry['gold_label']] for entry in data if entry['gold_label'] != '-'])
        eval(file_type)['label'] = g_labels
    print('extraction process was finished successfully!')
    return train, dev, test



def create_vocabulary(sentences, path):
    """
    creating the vocabulary by all sentences from train/dev/test
    :param sentences: sentences from train/dev/test
    :param path: path to text file with all embeddings
    :return: vocabulary
    """
    print('creating vocab..')

    word_dict = dict(); vocabulary = dict()
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''

    with open(path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                vocabulary[word] = np.fromstring(vec, sep=' ')

    print('vocabulary was created successfully!')
    return vocabulary



def save_vocabulary(path, vocab):
    """
    save vocabulary in pickle file
    :param path: data/vocabulary dorectory
    :param vocab: a given vocabulary
    """
    print('saving vocabulary..')
    with open(path, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('vocabulary was saved successfully!')


if __name__ == '__main__':

    # import all datasets
    train_data, dev_data, test_data = import_datasets(SNLI_PATH)

    # construct all sentences from train, dev and test files
    all_sentences = train_data['premise'] + train_data['hypothesis'] + \
          dev_data['premise'] + dev_data['hypothesis'] + \
          test_data['premise'] + test_data['hypothesis']

    # create and save vocabulary
    vocabulary = create_vocabulary(all_sentences, GLOVE_PATH)
    save_vocabulary(VOCABULARY_PATH, vocabulary)
