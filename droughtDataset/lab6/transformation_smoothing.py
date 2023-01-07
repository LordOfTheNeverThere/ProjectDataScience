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
file_tag="dtimeseries"


train = read_csv(f'../Data/TimeSeries/Aggregation/{file_tag}_Aggregation_D.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)


#### smoothing multivaried series

WIN_SIZE = (10, 50, 100, 150, 200, 250)
# for w in WIN_SIZE:
#     rolling_multi = train.rolling(window=w)
#     smooth_df_multi = rolling_multi.mean()
#     figure(figsize=(3*HEIGHT, HEIGHT))
#     plot_series(smooth_df_multi[target_multi], title=f'{target_multi} - Smoothing (win_size={w})', x_label=index_multi, y_label='value')
#     #plot_series(smooth_df_multi['lights'])
#     smooth_df_multi.to_csv(f'../Data/TimeSeries/Smoothing/{file_tag}_Aggregation_{w}.csv', index=True)
#     xticks(rotation = 45)
#     savefig(f'images/transformation/set2_data_smoothing_{target_multi}_{w}.png')
#     show()


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

test = read_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_test.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)


for w in WIN_SIZE:
    train = read_csv(f'../Data/TimeSeries/Smoothing/{file_tag}_Aggregation_{w}.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train["QV2M"])
    prd_tst = fr_mod.predict(test["QV2M"])

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test["QV2M"].values, prd_tst)

    print(w)
    print(eval_results)

    plot_evaluation_results(train["QV2M"].values, prd_trn, test["QV2M"].values, prd_tst, w)
    savefig(f'images/smoothing/set2_train_smoothing_{w}_results.png')
    plot_forecasting_series(train["QV2M"], test["QV2M"], prd_trn, prd_tst, f"Persistence Plot Forecasting Smoothing {w}", x_label=index_col, y_label=target)
    savefig(f'images/smoothing/set2_train_smoothing_{w}_plots.png')
