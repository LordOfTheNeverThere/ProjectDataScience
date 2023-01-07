import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, savefig, show
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

#### READ DATA

index_col = 'date'
target = 'QV2M'
file_tag="dtimeseries"
data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True)

#### FAZER OS TRAINING SETS

train, test = split_dataframe(data, trn_pct=0.70)

measure = 'R2'
flag_pct = False
eval_results = {}

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

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, "Simple Average Evaluation")
savefig(f'images/SimpleAverage/set2_SimpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, "Simple Average Plots", x_label=index_col, y_label=target)
savefig(f'images/SimpleAverage/set2_SimpleAvg_plots.png')