# %%
import numpy as np
import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sampleSubmission.csv')

# %%
tr_ftr = train.drop(['count'], axis=1)
te_ftr = train.drop(['count'], axis=1)
tr_tgt = train['count']
