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

#### READ DATA

index_col = 'date'
target = 'QV2M'
file_tag="dtimeseries"

#### FAZER OS TRAINING SETS

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
trnX, tstX, trnY, tstY = split_temporal_data(data, target, trn_pct=0.7)
train, test = split_dataframe(data, trn_pct=0.7)

train["QV2M"].to_csv(f'../Data/TimeSeries/SimpleAverage/{file_tag}_train.csv', index=True)
test["QV2M"].to_csv(f'../Data/TimeSeries/SimpleAverage/{file_tag}_test.csv', index=True)

### INITIALIZE

measure = 'R2'
flag_pct = False
eval_results = {}

#### SIMPLE AVERAGE

train = read_csv(f'../Data/TimeSeries/SimpleAverage/{file_tag}_train.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
test = read_csv(f'../Data/TimeSeries/SimpleAverage/{file_tag}_test.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

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

print(train.shape)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'Evaluation')
savefig(f'images/SimpleAverage/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, 'Forecasting Simple Average', x_label=index_col, y_label=target)
savefig(f'images/SimpleAverage/{file_tag}_simpleAvg_plots.png')

