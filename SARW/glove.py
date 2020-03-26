import numpy as np

def load_Glove(path_glove, word_list=None):
    if not isinstance(word_list, set):
        word_list = set()
    if isinstance(word_list, (bytes, str)):
        with open(word_list, encoding='utf-8') as f:
            for row in f:
                word_list.add(row.strip('\t').split()[0])
    word2vec = {}
    with open(path_glove, 'r', encoding='utf-8') as f:
        for row in f:
            word, vec = row.split(' ', 1)
            if word in word_list:
                word2vec[word] = np.fromstring(vec, sep=' ')
    return word2vec
