from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, sliding_window, shift_target, split_dataframe, split_temporal_data, plot_evaluation_results, PREDICTION_MEASURES, plot_forecasting_series
from matplotlib.pyplot import figure, savefig, show
from ds_charts import plot_evaluation_results, multiple_line_chart, multiple_bar_chart
from matplotlib.pyplot import subplots, Axes, gca
import matplotlib.dates as mdates
import config as cfg
from sklearn.base import RegressorMixin
from pandas import concat, Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy import ndarray, array
import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split


index_col = 'date'
target = 'QV2M'
data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, dayfirst=True)

print(data)
def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.7)


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

measure = 'R2'
flag_pct = False
eval_results = {}

fr_mod = RollingMeanRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'Rolling Mean Evaluation')
savefig(f'images/RollingMean/{file_tag}_rollingMean_eval.png')

plot_forecasting_series(train, test, prd_trn, prd_tst, 'Rolling Mean Plots', x_label=index_col, y_label=target)
savefig(f'images/RollingMean/{file_tag}_rollingMean_plots.png')
