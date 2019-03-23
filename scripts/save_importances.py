import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def save(models, tr_cols, time):
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
