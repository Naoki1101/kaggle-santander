from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def lr_train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lr_params, oof):

    logging.debug(lr_params)

    model = LogisticRegression(
        C=lr_params["C"],
        max_iter=lr_params["max_iter"],
        verbose=lr_params["verbose"],
        random_state=lr_params["random_state"],
        n_jobs=lr_params["n_jobs"]
    )
    model.fit(X_train, y_train)

    oof[X_valid.index] = model.predict_proba(X_valid)[:, 1]
    val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = roc_auc_score(y_valid, val_pred)
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred, model, val_auc, oof


def save_coef(models, tr_cols, time):
    feature_coef_df = pd.DataFrame()
    for fold_, model in enumerate(models):
        fold_coef_df = pd.DataFrame()
        fold_coef_df["Feature"] = tr_cols
        fold_coef_df["coef"] = np.abs(model.coef_.tolist()[0])
        fold_coef_df["fold"] = fold_ + 1
        feature_coef_df = pd.concat([feature_coef_df, fold_coef_df], axis=0)

    cols = (feature_coef_df[["Feature", "coef"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="coef", ascending=False).index)

    best_features = feature_coef_df.loc[feature_coef_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 100))
    sns.barplot(x="coef",
                y="Feature",
                data=best_features.sort_values(by="coef",
                                               ascending=False))
    plt.title('Logistic Regression Features (avg over folds)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('./importances/lr_coef_{0:%Y%m%d%H%M%S}.png'.format(time))


def save_features(oof, pred, overwrite=False):
    root = Path('./features/')
    train_path = root.joinpath('lr_train.feather')
    test_path = root.joinpath('lr_test.feather')
    if train_path.exists() and test_path.exists() and not overwrite:
        print('lr was skipped')
    else:
        pd.DataFrame(oof, columns=['lr_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['lr_pred']).to_feather(str(test_path))
