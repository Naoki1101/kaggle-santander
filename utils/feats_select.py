from scipy.stats import ks_2samp
import pandas as pd

def ks2samp(feats, tr, te, th=0.05):
    list_p_value =[]

    for col in feats:
        list_p_value.append(ks_2samp(te[col] , tr[col])[1])
    Se = pd.Series(list_p_value, index = feats).sort_values() 
    list_discarded = list(Se[Se < th].index)

    return list_discarded