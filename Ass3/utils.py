
STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

UNK = 'UNNNK'
VOCABULARY = {UNK}
TAGS = set()
UNIT_SUB_WORD = 3
W2I = {}    # words : index
I2W = {}    # index : words
T2I = {}    # tags : index
I2T = {}    # index : tags
C2I = {}    # character : index
P2I = {}    # prefixes : index
S2I = {}    # suffixes : index


def parse_data_from_file(filename):
    """
    parse date comprise of list of tuples.
    tuple -> {sentence : tags)
    :param filename: file name of train/dev data sets
    :return: list of tuples (for train/dev data sets)
    """
    sentence = []
    tags = []
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                data.append((sentence, tags))
                sentence = []
                tags = []
                continue
            word, tag = line.strip().split()
            sentence.append(word)
            tags.append(tag)
    return data


def create_dictionaries(data_train):
    """
    create all dictionaries as a global variables
    :param data_train: train data list
    :return:
    """
    global W2I, I2W, T2I, I2T, C2I, P2I, S2I
    # run through all train data set
    for sentence, tags in data_train:
        for word, tag in zip(sentence, tags):
            VOCABULARY.add(word)
            TAGS.add(tag)
    W2I = {w: i for i, w in enumerate(VOCABULARY)}
    I2W = {i: w for w, i in W2I.items()}
    T2I = {c: i for i, c in enumerate(TAGS)}
    I2T = {i: c for c, i in T2I.items()}
    C2I = {c: i for word in VOCABULARY for i, c in enumerate(word)}
    prefixes = {word[:UNIT_SUB_WORD] for word in VOCABULARY}
    suffixes = {word[-UNIT_SUB_WORD:] for word in VOCABULARY}
    P2I = {w: i for i, w in enumerate(prefixes)}
    S2I = {w: i for i, w in enumerate(suffixes)}


def create_unresolved_data(filename):
    """
    generating list of sentences where each sentence combined of examples/words to be tagged
    :param filename: file name of test data set
    :return: blind data
    """
    sentence = []
    untagged_data = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                untagged_data.append(sentence)
                sentence = []
                continue
            word = line.strip()
            sentence.append(word)
    return untagged_data


def create_results_file(pred_data, unresolved_data, results_filename):
    """
    set results into txt file.
    format: word  tag
    :param pred_data: data includes predictions
    :param unresolved_data: data without prediction
    :param results_filename: final file with predictions
    """
    results = []
    for sentence, pred_tags in zip(unresolved_data, pred_data):
        for word, tag in zip(sentence, pred_tags):
            results.append(str(word) + ' ' + str(tag))
        results.append(' ')
    with open(results_filename, 'w') as file:
        file.write('\n'.join(results))
