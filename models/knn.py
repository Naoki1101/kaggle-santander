from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path

import logging


def normalize(tr, te):
    whole = pd.concat([tr, te], axis=0)
    for col in whole.columns:
        whole[col] = minmax_scale(whole[col])

    tr_scaled = whole[:len(tr)]
    te_scaled = whole[len(tr):]

    return tr_scaled, te_scaled


def knn_train_and_predict(X_train, X_valid, y_train, y_valid, X_test, knn_params, oof):

    logging.debug(knn_params)

    model = KNeighborsRegressor(
        n_neighbors=knn_params['n_neighbors'],
        weights=knn_params['weights'],
        metric=knn_params['metric'],
        n_jobs=knn_params['n_jobs']
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_valid)
    oof[X_valid.index] = val_pred
    val_auc = roc_auc_score(y_valid, val_pred)
    y_pred = model.predict(X_test)

    return y_pred, model, val_auc, oof


def save_features(oof, pred, overwrite=False):
    root = Path('./features/')
    train_path = root.joinpath('knn_train.feather')
    test_path = root.joinpath('knn_test.feather')
    if train_path.exists() and test_path.exists() and not overwrite:
        print('knn was skipped')
    else:
        pd.DataFrame(oof, columns=['knn_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['knn_pred']).to_feather(str(test_path))
