# %% rolling mean

from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'ashrae'
index_col='timestamp'
target='meter_reading'
data = read_csv('data/time_series/ashrae_single.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

print(data.head())

# %%

train, test = split_dataframe(data, trn_pct=0.75)
measure = 'R2'
flag_pct = False
eval_results = {}

class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

fr_mod = RollingMeanRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{file_tag}_rollingMean_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_rollingMean_plots.png', x_label=index_col, y_label=target)