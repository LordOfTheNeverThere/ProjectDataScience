from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
index_multi = 'date'
target_multi = 'QV2M'
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, dayfirst=True)

#### differentiation multivaried series

diff_df_multi = data_multi.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df_multi[target_multi], title=f'{target_multi} - Differentiation', x_label=index_multi, y_label='value')
#plot_series(diff_df_multi['lights'])
xticks(rotation = 45)
savefig(f'images/transformation/set2_data_differentiation_{target_multi}.png')
show()