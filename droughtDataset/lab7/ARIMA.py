import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, HEIGHT, split_dataframe
from statsmodels.tsa.arima.model import ARIMA
from ds_charts import multiple_line_chart

#### READ DATA

index_col = 'date'
target = 'QV2M'
file_tag="dtimeseries"
data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col="date", usecols=["date", "QV2M"], sep=',', decimal='.', parse_dates=True, dayfirst=True)
train, test = split_dataframe(data, trn_pct=0.70)

#### ARIMA receives three mandatory parameters p, d, q
#### FIND THE BEST PARAMETERS

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
savefig(f'images/ARIMA/{file_tag}_ts_arima_study.png')
show()
print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')

#### PERFORMANCE

prd_trn = best_model.predict(start=0, end=len(train)-1)
prd_tst = best_model.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, '')
savefig(f'images/ARIMA/{file_tag}_ts_arima_eval_best.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'ARIMA Best Results Plots (p,d,q)=({best[0]}, {best[1]}, {best[2]})', x_label= str(index_col), y_label=str(target))
savefig(f'images/ARIMA/{file_tag}_ts_arima_plots_best.png')