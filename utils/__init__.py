import time
import pandas as pd
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print('[{name}] start'.format(name=name))
    yield
    print('[{name}] done in {end:.0f} s'.format(name=name, end=time.time()-t0))


def load_datasets(feats):
    dfs = [pd.read_feather('features/{f}_train.feather'.format(f=f)) for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather('features/{f}_test.feather'.format(f=f)) for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train
