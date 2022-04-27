import numpy as np
import pandas as pd
data= pd.read_csv("train_data.csv")
data
from scipy.stats import poisson, norm, rv_discrete, rv_continuous

def is_discrete(dist):

    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else: return isinstance(dist, rv_discrete)

def is_continuous(dist):

    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_continuous)
    else: return isinstance(dist, rv_continuous)
is_discrete(data[2])
