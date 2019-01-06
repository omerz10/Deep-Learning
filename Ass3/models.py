import numpy as np
import dynet as dy
import utils as ut


STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

LAYER_SIZE = 1
HIDDEN_DIM = 50
WORD_EMBED_DIM = 115
CHAR_EMBED_DIM = 20
SUBWORD_EMBED_DIM = 115


class BiLSTM(object):
    """
    initialize BiLSTM model with 1 layer
    """
    def __init__(self, in_dim, model):
        self.builders = [
            dy.LSTMBuilder(LAYER_SIZE, in_dim, HIDDEN_DIM, model),
            dy.LSTMBuilder(LAYER_SIZE, in_dim, HIDDEN_DIM, model)
        ]

    def __call__(self, sentence):
        f_in, b_in = [builder.initial_state() for builder in self.builders]
        f_output = f_in.transduce(sentence)
        b_output = b_in.transduce(reversed(sentence))
        concat = [dy.concatenate([f, b]) for f, b in zip(f_output, b_output)]
        return concat


class Amodel(object):
    """
    initialize Amodel which comprise of:
    - 2 layers BiLSTM
    - output to mlp with 1 output layer with 'softmax' as distribution function.
    """
    def __init__(self, w2i, t2i, i2t):
        self.w2i = w2i;  self.t2i = t2i;  self.i2t = i2t
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.E = self.model.add_lookup_parameters((len(self.w2i), WORD_EMBED_DIM))
        self.first_BiLSTM = BiLSTM(WORD_EMBED_DIM, self.model)
        self.second_BiLSTM = BiLSTM(2 * HIDDEN_DIM, self.model)
        self.W = self.model.add_parameters((len(self.t2i), 2 * HIDDEN_DIM))
        self.b = self.model.add_parameters(len(self.t2i))


    def __call__(self, sentence):
        """
        create embeddings by a given sentence.
        call the first and second BiLSTM models.
        each vector b will be fed into a linear layer followed by a softmax.
        :param sentence: given sentence
        :return:
        """
        dy.renew_cg()
        embeddings = [self.represent(word) for word in sentence]
        first_output = self.first_BiLSTM(embeddings)
        second_output = self.second_BiLSTM(first_output)
        return [dy.softmax(self.W * output_t + self.b) for output_t in second_output]


    def represent(self, word):
        """
        represent a given word.
        If not exist in dictionary return representation of 'ÚNK'
        :param word: word
        :return: word representation
        """
        if word in self.w2i:
            return self.E[self.w2i[word]]
        return self.E[self.w2i[ut.UNK]]


    def compute_loss(self, sentence, tags):
        """
        compute by cross-entropy loss
        :param sentence: sentence that will be predicted
        :param tags: tags of sentence
        :return: sum of all losses
        """
        s_loss = []
        s_tags = self(sentence)
        for word_tag, v_tag in zip(s_tags, tags):
            s_loss.append(-dy.log(dy.pick(word_tag, self.t2i[v_tag])))
        return dy.esum(s_loss)


    def compute_prediction(self, sentence):
        """
        for each word in sentence compute prediction of the tag
        :param sentence: sentence of words to be predicted
        :return: predicted tags of the sentence
        """
        s_tags = []
        s_pre_tags = self(sentence)
        for w_pre_tag in s_pre_tags:
            s_tags.append(self.i2t[np.argmax(w_pre_tag.value())])
        return s_tags


class Bmodel(Amodel):
    """
    initialize Bmodel comprise of char-level LSTM
    """
    def __init__(self, w2i, t2i, i2t, c2i):
        Amodel.__init__(self, w2i, t2i, i2t)
        self.c2i = c2i
        self.char_embed = self.model.add_lookup_parameters((len(self.c2i), CHAR_EMBED_DIM))
        self.char_builder = dy.LSTMBuilder(LAYER_SIZE, CHAR_EMBED_DIM, WORD_EMBED_DIM, self.model)


    def represent(self, word):
        """
        represent a given word by char embeddings
        :param word: given word
        :return: representation
        """
        char_init_state = self.char_builder.initial_state()
        char_embed = [self.char_embed[self.c2i[char]] for char in word]
        char_repr = char_init_state.transduce(char_embed)[-1]
        return char_repr


class Cmodel(Amodel):
    """
    initialize Cmodel comprise of embeddings(suffix and prefix) + subword representation
    """
    def __init__(self, w2i, t2i, i2t, p2i, s2i):
        Amodel.__init__(self, w2i, t2i, i2t)
        self.p2i = p2i
        self.s2i = s2i
        self.prefix_embed = self.model.add_lookup_parameters((len(self.p2i), SUBWORD_EMBED_DIM))
        self.suffix_embed = self.model.add_lookup_parameters((len(self.s2i), SUBWORD_EMBED_DIM))


    def represent(self, word):
        """
        represent a given word by adding word, prefix and suffix embeddings
        :param word:
        :return:
        """
        word_embed = Amodel.represent(self, word)
        suffix_embed = self.suffix_embed[self.s2i[self.get_suffix(word)]]
        prefix_embed = self.prefix_embed[self.p2i[self.get_prefix(word)]]
        repr = word_embed + prefix_embed + suffix_embed
        return repr


    def get_suffix(self, word):
        """
        get suffix of a given word.
        if word is not in dictionary, return suffix of UNK value
        :param word: given word
        :return: suffix
        """
        if word in self.w2i:
            return word[-ut.UNIT_SUB_WORD:]
        return ut.UNK[-ut.UNIT_SUB_WORD:]


    def get_prefix(self, word):
        """
        get prefix of a given word.
        if word is not in dictionary, return prefix of UNK value
        :param word: given word
        :return: prefix
        """
        if word in self.w2i:
            return word[:ut.UNIT_SUB_WORD]
        return ut.UNK[:ut.UNIT_SUB_WORD]


class Dmodel(Bmodel):
    """
    initialize Dmodel comprise of concating of word and char embeddings, followed by linear layer
    """
    def __init__(self, w2i, t2i, i2t, c2i):
        Bmodel.__init__(self, w2i, t2i, i2t, c2i)
        self.linear = self.model.add_parameters((WORD_EMBED_DIM, 2 * WORD_EMBED_DIM))
        self.bias = self.model.add_parameters(WORD_EMBED_DIM)


    def represent(self, word):
        """
        represent a given word by linear, concatenate and bias
        :param word:
        :return:
        """
        word_embeddings = Amodel.represent(self, word)
        char_embeddings = Bmodel.represent(self, word)
        lcb = self.linear * dy.concatenate([word_embeddings, char_embeddings]) + self.bias
        return lcb