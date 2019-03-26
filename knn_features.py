import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import StratifiedKFold

from utils.line_notification import send_message


class KnnFeatures():

    def _distance(self, a, b):
        dist = np.linalg.norm(b - a)
        return dist

    def _get_feat(self, data, X_train, y_train, class_index, k_index):
        inclass_X = X_train[y_train == class_index]
        distances = np.array([self._distance(a, data) for a in inclass_X])
        sorted_distances_index = np.argsort(distances)
        nearest_index = list(sorted_distances_index[0: (k_index + 1)])
        dist = np.sum(distances[nearest_index])
        return dist

    def knnExtract(self, X, y, k=1, folds=5):
        CLASS_NUM = len(set(y))
        res = np.empty((len(X), CLASS_NUM * k))
        kf = StratifiedKFold(n_splits=folds, shuffle=True)

        for fold_, (train_index, test_index) in enumerate(kf.split(X, y)):
            print("fold {}".format(fold_ + 1))
            start = time.time()
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]

            features = np.empty([0, len(X_test)])

            for class_index in range(CLASS_NUM):
                for k_index in range(k):
                    feat = np.array([np.apply_along_axis(
                        self._get_feat, 1,
                        X_test, X_train, y_train,
                        class_index, k_index
                    )])
                    features = np.append(features, feat, axis=0)
            res[test_index] = features.T
            message = """fold {f}\ntime {t}[min]""".format(f=fold_ + 1, t=(time.time() - start) / 60)
            self.send_message(message)
        return res


if __name__ == "__main__":
    print("===== data loading =====")
    df_train = pd.read_csv("./data/input/train.csv")
    df_test = pd.read_csv("./data/input/test.csv")
    print("===== data loaded =====")

    X = df_train[df_train.columns[2:]].values
    y = df_train.target.values

    print("===== create KNN Features start! =====")
    knn_features = KnnFeatures()
    newX = knn_features.knnExtract(X, y, k=1, folds=5)
    print("===== create KNN Features end! =====")
    print(datetime.datetime.now())
