import numpy as np
import DaPy as dp

exp_minmax = lambda x: np.exp(x - np.max(x))
denom = lambda x: 1.0 / np.sum(x)
def softmax(x):
    x = np.apply_along_axis(exp_minmax,1,x)
    denominator = np.apply_along_axis(denom,1,x) 
    if len(denominator.shape) == 1:
        denominator = denominator.reshape((denominator.shape[0],1))
    return x * denominator

def smoothmax(vector, axis=1):
    vector = vector - vector.max(axis=1)
    vector = np.exp(vector)
    return vector / np.abs(vector).sum(axis=1)


shape_of_postion_embedding = 50
def calPosEmbedding(lenth, shape=shape_of_postion_embedding):
    if lenth == 0:
        return np.array([])
    base = np.array([pos / np.power(10000, 2.0 * np.arange(shape) / shape) for pos in range(lenth)])
    base[:, 0::2] = np.sin(base[:, 0::2])
    base[:, 1::2] = np.cos(base[:, 1::2])
    return base

def calItemEmbedding(token, wordvec, shape=300):
    for method in (str.strip, str.lower, str.upper,
                   lambda x: x[0].upper() + x[1:].lower()):
        val = method(token)
        if val in wordvec:
            return wordvec[val]
    return np.zeros(shape)

def calItemsEmbedding(tokens, word2vec, shape=300):
    try:
        return np.vstack((calItemEmbedding(_, word2vec, shape) for _ in tokens))
    except ValueError:
        return np.array([])
