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

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, dayfirst=True)
index_multi = 'date'
target_multi = 'QV2M'
target = 'QV2M'
file_tag="dtimeseries"
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, dayfirst=True)
index_col = 'date'

#### aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

## multivaried series

granularity = ('D', 'W', 'M', 'Q', 'Y')
for j in range(len(granularity)):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_multi_df = aggregate_by(data_multi, index_multi, granularity[j])
    plot_series(agg_multi_df[target_multi], title=f'{target_multi} - {granularity[j]} values', x_label='timestamp', y_label='value')
    #plot_series(agg_multi_df['lights'])
    xticks(rotation = 45)
    savefig(f'images/transformation/set2_data_aggregation_{granularity[j]}.png')
    #show()

#### FAZER OS TRAINING SETS

trnX, tstX, trnY, tstY = split_temporal_data(data_multi, target_multi, trn_pct=0.70)
data = data_multi('QV2M', axis = 1) # see
data = data_multi.iloc[9:,:] # see
train, test = split_dataframe(data, trn_pct=0.70)

train.to_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_train.csv', index=False)
test.to_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_test.csv', index=False)

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

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/transformation/{file_tag}_persistence_eval.png')
savefig('images/transformation/set2_data_smoothing_results.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/transformation/{file_tag}_persistence_plots.png', x_label=index_col, y_label=target)
savefig(f'images/transformation/set2_data_smoothing_plots.png')