import numpy as np

STUDENT = {'name': 'Omer Zucker_Omer Wolf',
           'ID': '200876548_307965988'}

k = 5
W2V = {}

def words_to_vectors():
    """
    create dictionary of word : vectors
    :return: words to vectors dictionary
    """
    embedding = np.loadtxt("wordVectors")
    file = open('vocab', 'r')
    for index, line in enumerate(file):
        word = line.strip()
        W2V[word] = embedding[index]


def most_similar(word, k):
    """
    returns k most similar words of a given word
    :param word: word
    :param k: number of similar words
    :return: list of k most similar words
    """
    if word in W2V:
        u = W2V[word]
        values_d = {}
        # calculate distance for each vector
        for key, v in W2V.items():
            # skip same word
            if key == word:
                continue
            values_d[key] = np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))
        # sort by value in reverse
        sorted_values = sorted(values_d.items(), key=lambda t: t[1], reverse=True)
        res = [word for i, word in enumerate(sorted_values) if i < k]
        return res
    return []


if __name__ == '__main__':
    words_list = ['dog', 'england', 'john', 'explode', 'office']
    words_to_vectors()
    for i in range(len(words_list)):
        print(words_list[i] + ': ' + str(most_similar(words_list[i], k)) + '\n')

