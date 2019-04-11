import numpy as np


def p2rank(predicts):
    ranks = np.zeros(predicts.shape[0])
    rank = 0
    for i in np.argsort(predicts):
        ranks[i] = rank
        rank += 1
    return ranks
