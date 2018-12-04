import numpy

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}


UNK_SYMBOL = 'UUUNKKK'
FEATURES_1AND2 = 'F_start'
FEATURES_4AND5 = 'F_end'
WORDS = {UNK_SYMBOL}
TAGS = set()
W2I = {}    # word : index
I2W = {}    # index : word
T2I = {}    # tag : index
I2T = {}    # index : tag
P2I = {}    # prefix : index
I2P = {}    # index : prefix
S2I = {}    # suffix : index
I2S = {}    # index : suffix



def extract_words_and_tags(file_name):
    """
    extract features and labels from train file
    :param file_name:
    :return:
    """
    fp = open(file_name, 'r')
    for line in fp:
        if not line.strip():
            continue
        word, tag = line.strip().split()
        WORDS.add(word)
        TAGS.add(tag)

def create_dicts():
    """
    create dictionaries for words and tags
    :return: none
    """
    W2I.update({w: i for i, w in enumerate(WORDS)})
    I2W.update({i: w for w, i in W2I.items()})
    T2I.update({t: i for i, t in enumerate(TAGS)})
    I2T.update({i: t for t, i in T2I.items()})


def create_additional_dict(subwords_size):
    """
    create dictionaries of prefixes and suffixes
    :param subwords_size: number of features to be determine in the window
    :return: none
    """
    prefixes = {w[:subwords_size] for w in  WORDS}
    suffixes = {w[-subwords_size:] for w in WORDS}
    P2I.update({p: i for i, p in enumerate(prefixes)})
    I2P.update({i: p for p, i in P2I.items()})
    S2I.update({s: i for i, s in enumerate(suffixes)})
    I2S.update({i: s for s, i in S2I.items()})


def create_features_and_labels_vec(file_name):
    """
    create sentences by words within the dictionaries and convert them into features and labels vectors
    :param file_name: any train or dev file
    :return: features and labels vectors
    """
    single_sentence, sentences, features_vec, labels_vec = [], [] , [] , []
    # create all sentences from a given file by adding 'dummies' features (5 features at total)
    file = open(file_name, 'r')
    for line in file:
        if not line.strip():
            sentence_start = [(FEATURES_1AND2, FEATURES_1AND2)] * 2
            sentence_end = [(FEATURES_4AND5, FEATURES_4AND5)] * 2
            sentences.append( sentence_start + single_sentence + sentence_end)
            single_sentence = []
            continue
        word, tag = line.strip().split()
        single_sentence.append((word, tag))
    # create numeric vectors of both features and labels
    for s in sentences:
        for i, (word, tag) in enumerate(s[2:-2], 2):
            words = [s[i - 2][0], s[i - 1][0], word, s[i + 1][0], s[i + 2][0]]
            features_vec.append(words_to_vec(words))
            labels_vec.append(T2I[tag])
    return features_vec, labels_vec


def create_test_features_vec(test_fname):
    """
    extract words as features for test file
    :param test_fname: name of test file
    :return: vectors of features
    """
    single_sentence, sentences, features_vec = [], [] , []
    # create all sentences from a given test file by adding 'dummies' features (3 features at total)
    file = open(test_fname, 'r')
    for line in file:
        if not line.strip():
            sentence_start = [FEATURES_1AND2] * 2
            sentence_end = [FEATURES_4AND5] * 2
            sentences.append( sentence_start + single_sentence + sentence_end)
            single_sentence = []
            continue
        word = line.strip()
        single_sentence.append(word)
    # create numeric vectors of both features and labels
    for s in sentences:
        for i, word in enumerate(s[2:-2], 2):
            words = [s[i - 2][0], s[i - 1][0], word, s[i + 1][0], s[i + 2][0]]
            features_vec.append(words_to_vec(words))
    return features_vec


def words_to_vec(words_list):
    """
    keep words as vectors in dictionaries and return the vectors
    :param words_list: list of words
    :return: vectors of words
    """
    vectors = []
    for word in words_list:
        if word in WORDS:
            vectors.append(W2I[word])
        elif word.lower() in WORDS:
            vectors.append(W2I[word.lower()])
        else:
            vectors.append(W2I[UNK_SYMBOL])
    return vectors
