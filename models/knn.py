from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score


def knn_train_and_predict(X_train, X_valid, y_train, y_valid, X_test, knn_params):

    model = KNeighborsRegressor(
        n_neighbors=knn_params['n_neighbors'],
        weights=knn_params['weights'],
        metric=knn_params['metric'],
        n_jobs=knn_params['n_jobs']
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_valid)
    val_auc = roc_auc_score(y_valid, val_pred)
    y_pred = model.predict(X_test)

    print("valid_1's auc: %.4f" % val_auc)

    return y_pred, model
