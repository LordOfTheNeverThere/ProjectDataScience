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

#### READ DATA

index_col = 'date'
index_multi = 'date'
target = 'QV2M'
target_multi = 'QV2M'
file_tag="dtimeseries"
data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, dayfirst=True)

#### FAZER OS TRAINING SETS

trnX, tstX, trnY, tstY = split_temporal_data(data_multi, target_multi, trn_pct=0.70)
# #data = data_multi('QV2M', axis = 1) # see
# #data = data_multi.iloc[9:,:] # see
# train, test = split_dataframe(data, trn_pct=0.70)
#
# train.to_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_train.csv', index=False)
# test.to_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_test.csv', index=False)

# #### AGGREGATION
#
# train = read_csv(f'../Data/TimeSeries/TrainTest/{file_tag}_train.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)
#
# def aggregate_by(data: Series, index_var: str, period: str):
#     index = data.index.to_period(period)
#     agg_df = data.copy().groupby(index).mean()
#     agg_df[index_var] = index.drop_duplicates().to_timestamp()
#     agg_df.set_index(index_var, drop=True, inplace=True)
#     return agg_df
#
# ## MULTIVARIED SERIES
#
# granularity = ('D', 'W', 'M', 'Q', 'Y')
# for g in granularity:
#     figure(figsize=(3*HEIGHT, HEIGHT))
#     agg_multi_df = aggregate_by(train, index_multi, g)
#     agg_multi_df.to_csv(f'../Data/TimeSeries/Aggregation/{file_tag}_Aggregation_{g}.csv', index=True)
#     plot_series(agg_multi_df[target_multi], title=f'{target_multi} - {g} values', x_label='timestamp', y_label='value')
#     #plot_series(agg_multi_df['lights'])
#     xticks(rotation = 45)
#     savefig(f'images/aggregation/set2_train_aggregation_{g}.png')
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


granularity = ('D', 'W', 'M', 'Q', 'Y')

for g in granularity:
    train = read_csv(f'../Data/TimeSeries/Aggregation/{file_tag}_Aggregation_{g}.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train["QV2M"])
    prd_tst = fr_mod.predict(test["QV2M"])

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test["QV2M"].values, prd_tst)

    print(g)
    print(eval_results)

<<<<<<< HEAD
    plot_evaluation_results(train["QV2M"].values, prd_trn, test["QV2M"].values, prd_tst, f'images/aggregation/{file_tag}_{g}_persistence_eval.png')
    savefig(f'images/aggregation/set2_train_aggregation_{g}_results.png',title = f"Persistence Results Aggregation {g}")
    plot_forecasting_series(train["QV2M"], test["QV2M"], prd_trn, prd_tst, f'images/aggregation/{file_tag}_{g}_persistence_plots.png', x_label=index_col, y_label=target, title = f"Persistence Results Aggregation {g}")
    savefig(f'images/aggregation/set2_train_aggregation_{g}_plots.png')
=======
    plot_evaluation_results(train["QV2M"].values, prd_trn, test["QV2M"].values, prd_tst, g)
    savefig(f'images/aggregation/set2_train_aggregation_{g}_results.png')
    plot_forecasting_series(train["QV2M"], test["QV2M"], prd_trn, prd_tst, f"Persistence Plot Forecasting Aggregation {g}", x_label=index_col, y_label=target)
    savefig(f'images/aggregation/set2_train_aggregation_{g}_plots.png')

>>>>>>> c0257409b4d644724201b7cbd0d93978e66de4fd
