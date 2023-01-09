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

# %% 

from statsmodels.tsa.arima.model import ARIMA

pred = ARIMA(train, order=(2, 0, 2))
model = pred.fit(method_kwargs={'warn_convergence': False})
model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))


# %% best model - thy are designed for stationary time series or non stationarity transformed to stationary


from matplotlib.pyplot import subplots, show, savefig
from ds_charts import multiple_line_chart
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

measure = 'R2'
flag_pct = False
last_best = -100
best = ('',  0, 0.0)
best_model = None

d_values = (0, 1, 2)
params = (1, 2, 3, 5)
ncols = len(d_values)

fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)

for der in range(len(d_values)):
    d = d_values[der]
    values = {}
    for q in params:
        yvalues = []
        for p in params:
            pred = ARIMA(train, order=(p, d, q))
            model = pred.fit(method_kwargs={'warn_convergence': False})
            prd_tst = model.forecast(steps=len(test), signal_only=False)
            yvalues.append(PREDICTION_MEASURES[measure](test,prd_tst))
            if yvalues[-1] > last_best:
                best = (p, d, q)
                last_best = yvalues[-1]
                best_model = model
        values[q] = yvalues
    multiple_line_chart(
        params, values, ax=axs[0, der], title=f'ARIMA d={d}', xlabel='p', ylabel=measure, percentage=flag_pct)

savefig('images/arima_study.png')

print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')

# Best results achieved with (p,d,q)=(1, 1, 3) ==> measure=0.01
# %% best model in practice

from statsmodels.tsa.arima.model import ARIMA
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

prd_trn = best_model.predict(start=0, end=len(train)-1)
prd_tst = best_model.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'Eval ARIMA')
savefig('images/arima_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, 'Plots ARIMA', x_label= 'Date', y_label='Glucose')
savefig('images/arima_plots.png')

# %%
