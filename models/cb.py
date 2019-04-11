from catboost import Pool, CatBoostClassifier

import logging
from logs.logger import log_evaluation

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, cb_params, oof):

    logging.debug(cb_params)

    cb_train = Pool(X_train, label=y_train)
    cb_valid = Pool(X_valid, label=y_valid)

    cb = CatBoostClassifier(
        loss_function=cb_params['loss_function'],
        eval_metric=cb_params['eval_metric'],
        learning_rate=cb_params['learning_rate'],
        iterations=cb_params['iterations'],
        random_seed=cb_params['random_seed'],
        od_type=cb_params['od_type'],
        depth=cb_params['depth'],
        early_stopping_rounds=cb_params['early_stopping_rounds']
    )
    model = cb.fit(
        cb_train,
        eval_set=cb_valid,
        use_best_model=True,
        verbose_eval=200,
        plot=False
    )

    oof[X_valid.index] = model.predict_proba(X_valid)[:, 1]
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred, model, oof


def save_importances(models, tr_cols, time):
    feature_importance_df = pd.DataFrame()
    for fold_, model in enumerate(models):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = tr_cols
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 100))
    sns.barplot(x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title(' Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('./importances/cb_importances_{0:%Y%m%d%H%M%S}.png'.format(time))


def save_features(oof, pred, overwrite=False):
    root = Path('./features/')
    train_path = root.joinpath('cb_train.feather')
    test_path = root.joinpath('cb_test.feather')
    if train_path.exists() and test_path.exists() and not overwrite:
        print('cb was skipped')
    else:
        pd.DataFrame(oof, columns=['cb_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['cb_pred']).to_feather(str(test_path))
