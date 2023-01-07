from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show
from ds_charts import plot_evaluation_results, multiple_line_chart
from matplotlib.pyplot import subplots, Axes, gca
import matplotlib.dates as mdates
import config as cfg
from sklearn.base import RegressorMixin
from pandas import concat, Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ds_charts import multiple_bar_chart
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy import ndarray, array
from ts_functions import sliding_window, shift_target, split_dataframe, split_temporal_data, plot_evaluation_results, PREDICTION_MEASURES, plot_forecasting_series
import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

index_col = 'date'
index_multi = 'date'
target = 'QV2M'
target_multi = 'QV2M'
#file_tag="dtimeseries"
file_tag="data"

data = read_csv(f'../Data/TimeSeries/Smoothing/TrainSmooth100+Test.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
#train = read_csv(f'../Data/TimeSeries/Smoothing/{file_tag}_Smoothing_100.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

#### differentiation multivaried series
#ORIGINAL
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(data[target_multi], title=f'{target_multi} - Differentiation 0', x_label=index_multi, y_label='value')
# xticks(rotation = 45)
# savefig(f'images/differentiation/set2_TrainSmooth100+Test_differentiation_0.png')
# show()

# diff1_df_multi = data.diff()
# diff1_df_multi.to_csv(f'../Data/TimeSeries/Differentiation/{file_tag}_Differentiation_1.csv', index=True)
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(diff1_df_multi[target_multi], title=f'{target_multi} - Differentiation 1', x_label=index_multi, y_label='value')
# xticks(rotation = 45)
# savefig(f'images/differentiation/set2_TrainSmooth100+Test_differentiation_1.png')
# show()

# diff2_df_multi = diff1_df_multi.diff()
# diff2_df_multi.to_csv(f'../Data/TimeSeries/Differentiation/{file_tag}_Differentiation_2.csv', index=True)
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(diff2_df_multi[target_multi], title=f'{target_multi} - Differentiation 2', x_label=index_multi, y_label='value')
# xticks(rotation = 45)
# savefig(f'images/differentiation/set2_TrainSmooth100+Test_differentiation_2.png')
# show()

#### CLASSIFIER
measure = 'R2' #we have 3 options
flag_pct = False
eval_results = {}

class PersistenceRegressor (RegressorMixin):
        def __init__(self):
            super().__init__()
            self.last = 0

        def fit(self, X: DataFrame):
            self.last = X.iloc[-1,0]
            print(self.last)

        def predict(self, X: DataFrame):
            prd = X.shift().values
            prd[0] = self.last
            return prd

for w in range(3):
    train = read_csv(f'../Data/TimeSeries/Differentiation/diff{w}train.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
    test = read_csv(f'../Data/TimeSeries/Differentiation/diff{w}test.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train["QV2M"])
    prd_tst = fr_mod.predict(test["QV2M"])

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test["QV2M"].values, prd_tst)

    print(w)
    print(eval_results)

    plot_evaluation_results(train["QV2M"].values, prd_trn, test["QV2M"].values, prd_tst, w)
    savefig(f'images/differentiation/set2_TrainSmooth100+Test_diff{w}_results.png')
    plot_forecasting_series(train["QV2M"], test["QV2M"], prd_trn, prd_tst, f"Persistence Plot Forecasting Differentiation {w}", x_label=index_col, y_label=target)
    savefig(f'images/differentiation/set2_TrainSmooth100+Test_diff{w}_plots.png')
