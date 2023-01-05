#%%-Imports
import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

#%%- Load Dataset
file_tag = 'diabetesV0.1'
index_col = 'Date'
target = 'Glucose'

data = read_csv('dummy.csv')
data['Date'] = pd.to_datetime(data['Date'], format = "%d/%m/%Y %H:%M")
data = data.set_index('Date') ## Droping Index col


train, test = split_dataframe(data, trn_pct=0.75)

measure = 'R2'
flag_pct = False
eval_results = {}

#%% Redifining Simples regressor Class
class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd = len(X) * [self.mean]
        return prd


# %% Simple Avg
fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values,
                        prd_tst, f'images/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_simpleAvg_plots.png', x_label=index_col, y_label=target)
