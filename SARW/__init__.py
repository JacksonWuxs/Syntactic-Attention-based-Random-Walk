from itertools import repeat
from pickle import dump, load
from time import clock
from collections import defaultdict
from random import random

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier

from .glove import load_Glove
from .parser import load_spacy_parser, clean_text
from .calculator import calItemsEmbedding, calPosEmbedding

__version__ = '0.1.1'
__author__ = 'Xuansheng Wu (wuxsmail@163.com)'


class SARW(object):
    
    name = 'SARW'
    
    def __init__(self, word2vec={}, freq={}, clf=None, tokenize=None):
        self.freq = freq # dict of word frequency
        self._size = sum(self.freq.values(), 1.0)
        self._clf = clf # classifier of word-word relation
        self._w2v = word2vec # word embedding dict
        self._nft = tokenize if tokenize is not None else 'en_core_web_md' # name of tokenize
        self.set_tokenizer(self._nft)

    def __setstate__(self, pkl):
        self.freq = pkl['freq']
        self._size = sum(self.freq.values(), 1.0)
        self._clf = pkl['clf']
        self._nft = pkl['nft']
        try:
            self.set_tokenizer(self._nft)
        except ImportError:
            print(' - Loading tokenizer failed, please use SARW.set_tokenizer() to reset the tokenizer')

    def __getstate__(self):
        return {'freq': self.freq,
                'clf': self._clf,
                'nft': self._nft}

    def load_freq(self, path):
        self.freq = defaultdict(int)
        with open(path, encoding='utf-8') as f:
            for row in f:
                word, freq = row.strip(' ').split()
                self.freq[word] = float(freq)
        self._size = sum(self.freq.values(), 1.0)
        self.freq['.'] = self.freq[','] = max(self.freq.values())
        return self

    def load_word2vec(self, path):
        self._w2v = load_Glove(path, set(self.freq.keys()))

    def set_tokenizer(self, name_of_tokenizer='en_core_web_md'):
        if name_of_tokenizer is None:
            self._tokenize = lambda text: print("the model does't contain tokenizers, please input tokenized sentence")
             
        if name_of_tokenizer == 'nltk':
            from nltk.tokenize import word_tokenize
            self._tokenize = lambda text: word_tokenize(clean_text(text))
            
        if name_of_tokenizer == 'spacy' or name_of_tokenizer.startswith('en_core_web_md'):
            from spacy import load
            tkns = load(name_of_tokenizer).tokenizer
            self._tokenize = lambda text: [_.text for _ in tkns(clean_text(text))]
            
        if name_of_tokenizer == 'jieba':
            from jieba import cut
            self._tokenize = lambda text: list(cut(text))
            

    def fit(self, corpus, spacy_parser='en_core_web_md'):
        start = clock()

        print(' - begin to train attention')
        trainX, testX = [], []
        trainY, testY = [], []
        parse = load_spacy_parser(spacy_parser)
        for num, row in enumerate(corpus, 1):
            if num % 10000 == 0:
                spent = clock() - start
                speed = num / spent
                print('   Done: %d (%.1f r/sec)' % (num, speed))
                
            tokens = parse(row)
            text_tokens = [_.text for _ in tokens]
            glove = calItemsEmbedding(text_tokens, self._w2v)
            posve = calPosEmbedding(len(tokens), 300)
            vector = glove + posve
    
            for i, token in enumerate(tokens):
                for j, pair in enumerate(tokens):
                    y = 1 if token.head.i == j else 0
                    if y == 1 or random() <= 0.1:
                        x = np.hstack([vector[i], vector[j]])
                        if random() > 0.9:
                            testX.append(x)
                            testY.append(y)
                        else:
                            trainX.append(x)
                            trainY.append(y)
        
        self._clf = MLPClassifier(verbose=1,
                solver='sgd', early_stopping=True,
                learning_rate='adaptive', learning_rate_init=0.05,
                hidden_layer_sizes=(64, 32, 32, 16), 
                max_iter=5000, activation='relu'
            )
        self._clf.fit(np.vstack(trainX), np.vstack(trainY))
        prd = self._clf.score(testX, testY)
        avg = 1.0 - np.mean(testY)
        print('   * Attention score: %.4f/%.4f' % (prd, avg))

    def fine_tune(self, corpus):
        counter = defaultdict(int)
        start = clock()
        print(' - begin to fine tune')
        for num, row in enumerate(corpus, 1):
            tokens = spacy_tokenize(row)
            for token in tokens:
                counter[token] += 1
            if num % 10000 == 0:
                spent = clock() - start
                speed = num / spent
                print('    Done: %d (%.1f r/sec)' % (num, speed))
        for key, freq in counter.items():
            self.freq[key]+= freq
            self._size += freq

    def transform(self, texts, a=1000000, b=0.3, m=10):
        func = lambda text: self._transform_once(text, a, b)
        vectors = np.vstack(map(func, texts))
        return self._rm_pc(vectors, m)

    def _transform_once(self, tokens, alpha, beta):
        if isinstance(tokens, (str, bytes)):
            tokens = self._tokenize(tokens)
        length = len(tokens)

        glove = calItemsEmbedding(tokens, self._w2v, shape=300)
        posve = calPosEmbedding(len(tokens), 300)
        vector = glove + posve

        if length == 0:
            return np.zeros(600)
        if length == 1:
            return np.hstack([glove, glove])
        
        A = self._get_am(glove, length, posve, beta)
        P = self._get_p(tokens, self.freq, self._size)
        W = alpha * P + A.sum(axis=1)
        H = (A.T / W).T
        V = self._get_MSU_vectors(length, glove)
        return H.flatten().dot(V)
    
    def _get_MSU_vectors(self, l, v):
        func = lambda r: np.hstack([np.vstack(repeat(r, l)), v])
        return np.vstack(map(func, v))
    
    def _get_am(self, v, l, p, beta):
        '''This function shows how does attention mechanism work'''
        X = self._get_MSU_vectors(l, v)
        a = self._clf.predict_proba(X)[:, 1]
        return a.reshape((l, l)) + beta * p.dot(p.T)
    
    def _get_p(self, tokens, freq, total):
        '''This function shows how to introduce smoothing term'''
        func = lambda t: freq.get(t, freq.get(t.lower(), 1))
        return np.array(tuple(map(func, tokens))) / total
        
    def _rm_pc(self, vectors, m):
        '''This function shows how to remove principal components'''
        svd = TruncatedSVD(n_components=m).fit(vectors)
        comps, singvals = svd.components_, svd.singular_values_
        singular_value_ratio = singvals / singvals.sum()
        for svr, comp in zip(singular_value_ratio, comps):
            comp = comp.reshape((1, vectors.shape[1]))
            vectors -= svr * vectors.dot(comp.transpose()) * comp
        return vectors


def trainSARW(corpus, w2v_path, save_path='output/sarw.pkl', freq_path='SARW/data/enwiki_vocab_min200.txt'):
    GLOVE = load_Glove(w2v_path, path_freq)
    sarw = SARW(word2vec=GLOVE)
    sarw.fit(corpus)
    sarw.load_freq(freq_path)
    with open(save_path, 'wb') as output:
        dump(sarw, output)
    return sarw

def loadSARW(w2v_path, save_path='output/sarw.pkl', tokenizer=None):
    with open(path, 'rb') as model:
        sarw = load(model)
    sarw.load_word2vec(glove_path)
    if tokenizer is not None:
        sarw.set_tokenizer(tokenizer)
    return sarw


if __name__ == '__main__':
    sarw = trainSARW('corpus/wiki.txt')
    sarw = loadSARW('output/sarw.pkl')
    texts = [
             u'The girl eats her bread',
             u'The beautiful girl eats her bread',
             u'The girl eats her delicious bread'
             ]
    embeddings = sarw.transform(texts, a=10000, b=5.0)
    visualization(texts, spacy_tokenize, embeddings)
