import pandas as pd

alg = [
    'tsne',
    'knn'
]

target = [
    'train',
    'test'
]

extension = 'feather'


for a in alg:
    for t in target:
        file_name = a + "_" + t + "." + extension
        df_ = pd.read_feather("./features/" + file_name)
        for i in range(df_.shape[1]):
            path = './features/{a}_{n}_{t}.feather'.format(a=a, n=i + 1, t=t)
            pd.DataFrame(df_.iloc[:, i]).to_feather(path)
