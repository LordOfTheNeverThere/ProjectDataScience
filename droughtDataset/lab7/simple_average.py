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
from ts_functions import 
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

#### READ

train = read_csv(f'../Data/TimeSeries/Differentiation/{file_tag}_Differentiation_0.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
test = read_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_test.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

#### INITIALIZE

measure = 'R2'
flag_pct = False
eval_results = {}

#### SIMPLE AVERAGE

#### CLASSIFIER

class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

#### PLOTS AND RESULTS

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train["QV2M"].values, prd_trn, test["QV2M"].values, prd_tst, 'Evaluation')
savefig(f'images/SimpleAverage/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train["QV2M"], test["QV2M"], prd_trn, prd_tst, 'Forecasting Simple Average', x_label=index_col, y_label=target)
savefig(f'images/SimpleAverage/{file_tag}_simpleAvg_plots.png')

