import lightgbm as lgb
import logging

from logs.logger import log_evaluation

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params, oof):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    logging.debug(lgbm_params)

    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=100)]

    num_round = 20000
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=num_round,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=200,
        early_stopping_rounds=200,
        callbacks=callbacks
    )

    oof[X_valid.index] = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return y_pred, model, oof


def remove_row(df, target, idx):
    drop_row = []
    for i in idx:
        if target.iloc[i] == 0:
            drop_row.append(i)
    logging.debug(len(drop_row))
    df = df.drop(drop_row, axis=0).reset_index(drop=True)
    target = target.drop(drop_row, axis=0).reset_index(drop=True)
    return df, target



def save_importances(models, tr_cols, time):
    feature_importance_df = pd.DataFrame()
    for fold_, model in enumerate(models):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = tr_cols
        fold_importance_df["importance"] = model.feature_importance()
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
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('./importances/lgbm_importances_{0:%Y%m%d%H%M%S}.png'.format(time))


def save_features(oof, pred, overwrite=False):
    root = Path('./features/')
    train_path = root.joinpath('lgbm_train.feather')
    test_path = root.joinpath('lgbm_test.feather')
    if train_path.exists() and test_path.exists() and not overwrite:
        print('lgbm was skipped')
    else:
        pd.DataFrame(oof, columns=['lgbm_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['lgbm_pred']).to_feather(str(test_path))
