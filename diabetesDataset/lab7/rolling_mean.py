# %% iniciation
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from pandas import read_csv, DataFrame, to_datetime
from matplotlib.pyplot import figure, subplots, savefig
from ts_functions import HEIGHT, split_dataframe

data = read_csv('preTransformationsGlucose.csv', usecols=['Date', 'Glucose'])
data['Date'] = to_datetime(data['Date'], format = "%Y/%m/%d %H:%M")
data = data.set_index('Date')

def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.75)

# %% rolling mean study

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

fr_mod = RollingMeanRegressor(2)
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

measure = 'R2'
flag_pct = False
eval_results = {}

eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'Eval Rolling Mean')
savefig('images/rolling_mean_2_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, 'Plot Rolling Mean', x_label='Date', y_label='Value')
savefig('images/rolling_mean_2_plots.png')
# %% rolling mean study
import matplotlib.pyplot as plt

r2_train = []
r2_test = []

for x in range(2,80):
    print(x)

    fr_mod = RollingMeanRegressor(x)
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    r2_train.append(PREDICTION_MEASURES['R2'](train, prd_trn))
    r2_test.append(PREDICTION_MEASURES['R2'](test, prd_tst))

fig, ax = plt.subplots()
ax.plot(range(2,80), r2_train, label='Train values')
ax.plot(range(2,80), r2_test, label='Test values')

legend = ax.legend(loc='upper center', fontsize='x-small')
plt.savefig('images/rollling_mean_study')
# %%
