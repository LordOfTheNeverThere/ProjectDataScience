# %% Imports
from pandas import DataFrame
from ts_functions import HEIGHT, split_dataframe, plot_series,PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from sklearn.base import RegressorMixin
from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
import pandas as pd

# %% Get Glucose Data read_csv
data = read_csv('glucose_after_smoothing.csv')

data.drop(index=range(0,9), inplace=True)
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
data = data.set_index('Date')  # Droping Index col

# %% 1st Differentiation  The role of this technique is to adress non stationary data, removing the influence of the trend I think

dataDiff1 = data.diff()
dataDiff1 = dataDiff1.iloc[1:, :]
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(dataDiff1['Glucose'], title='Diabetes - Differentiation', y_label='Glucose')
plot_series(data['Insulin'], y_label='Insulin')
xticks(rotation=45)
show()

# %% 2nd Differentiation  The role of this technique is to adress non stationary data, removing the influence of the trend I think

dataDiff2 = dataDiff1.diff()
dataDiff2 = dataDiff2.iloc[1:, :]
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(dataDiff2['Glucose'],
            title='Diabetes - Differentiation', y_label='Glucose')
plot_series(data['Insulin'], y_label='Insulin')

xticks(rotation=45)
show()


# %% Split
glucoseData = data.drop(columns=['Insulin'])
glucoseDataDiff1=dataDiff1.drop(columns=['Insulin'])
glucoseDataDiff2=dataDiff2.drop(columns=['Insulin'])

train, test = split_dataframe(glucoseData, trn_pct=0.75)
trainDiff1, testDiff1 = split_dataframe(glucoseDataDiff1, trn_pct=0.75)
trainDiff2, testDiff2 = split_dataframe(glucoseDataDiff2, trn_pct=0.75)

# %% persistance test Result-> NOTE:Best Diff is the first order one

measure = 'R2'
flag_pct = False
eval_results = {}


class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1, 0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd


# Zero order regressor
fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](
    test.values, prd_tst)
print(eval_results)

# %%

plot_evaluation_results(train.values, prd_trn, test.values,
                        prd_tst, 'Original Eval Results')
savefig('images/transformation/diff0Eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst,
                        'Original Forecasting', x_label='Date', y_label='Glucose')
savefig('images/transformation/diff0Plots.png')

## First order regressor
fr_mod = PersistenceRegressor()
fr_mod.fit(trainDiff1)
prd_trn = fr_mod.predict(trainDiff1)
prd_tst = fr_mod.predict(testDiff1)



# %%
eval_results['Persistence'] = PREDICTION_MEASURES[measure](
    testDiff1.values, prd_tst)
print(eval_results)

# %%

plot_evaluation_results(trainDiff1.values, prd_trn, testDiff1.values,
                        prd_tst, '1stOrder Eval Results')
savefig('images/transformation/diff1Eval.png')
plot_forecasting_series(trainDiff1, testDiff1, prd_trn, prd_tst,
                        '1stOrderDiff Forecasting', x_label='Date', y_label='Glucose')
savefig('images/transformation/diff1Plots.png')

## Second Regressor
fr_mod = PersistenceRegressor()
fr_mod.fit(trainDiff2)
prd_trn = fr_mod.predict(trainDiff2)
prd_tst = fr_mod.predict(testDiff2)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](
    testDiff2.values, prd_tst)
print(eval_results)

# %%

plot_evaluation_results(trainDiff2.values, prd_trn, testDiff2.values,
                        prd_tst, '2ndOrder Eval Results')
savefig('images/transformation/diff2Eval.png')
plot_forecasting_series(trainDiff2, testDiff2, prd_trn, prd_tst,
                        '2ndOrderDiff Forecasting', x_label='Date', y_label='Glucose')
savefig('images/transformation/diff2Plots.png')


#%% Save best 1st Order diff
dataDiff1.to_csv('preForecastingGlucoseDiff1.csv')

# %%
