# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_pinball_loss, mean_squared_error

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sampleSubmission.csv')

print('train', train.shape)
print('test', test.shape)
# %%
tr_tgt = train[['count']]

# %%
class Regressor:
    
    def __init__(self, tr_ftr, te_ftr, tr_tgt):
        self.tr_ftr = tr_ftr
        self.te_ftr = te_ftr
        self.tr_tgt = tr_tgt

    def _do_preprocessing(self):
        scaler = StandardScaler()
        self.tr_ftr = scaler.fit_transform(self.tr_ftr)
        self.te_ftr = scaler.transform(self.te_ftr)

        return self.tr_ftr, self.te_ftr

    def do_visualize(self, target, predict):
        fig = pgo.Figure(layout=dict(width=1000, height=600))
        fig.add_trace(pgo.Scatter(x=np.arange(len(target)), y=target, name='true'))
        fig.add_trace(pgo.Scatter(x=np.arange(len(target)), y=predict, name='pred'))        
        fig.show()

    def do_regression(self):
        tr_ftr, te_ftr = self._do_preprocessing()
        reg = RandomForestRegressor(n_jobs=-1)
        reg.fit(tr_ftr, self.tr_tgt)
        predict = reg.predict(te_ftr)

        return predict

    def do_calc_metric(self, target, predict):
        mae = mean_absolute_error(target, predict)
        mape = mean_absolute_percentage_error(target, predict)
        mpl = mean_pinball_loss(target, predict)
        mse = mean_squared_error(target, predict)
        print(f'MAE : {mae}')
        print(f'MAPE : {mape}')
        print(f'PINBALL : {mpl}')
        print(f'MSE : {mse}')

# %%

regressor = Regressor()