from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data, x_label='timestamp', y_label='value', title='Drought Forecasting Original')
xticks(rotation = 45)
savefig('images/transformation/set2_data_transformation_1.png')
show()

#### time series transformation

from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

index_multi = 'date'
target_multi = 'QV2M'
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='value', title=target_multi)
#plot_series(data_multi['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_transformation_2.png')
show()

#### smoothing

# WIN_SIZE = 10
# rolling = data.rolling(window=WIN_SIZE)
# smooth_df = rolling.mean()
# figure(figsize=(3*HEIGHT, HEIGHT/2))
# plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

# WIN_SIZE = 100
# rolling = data.rolling(window=WIN_SIZE)
# smooth_df = rolling.mean()
# figure(figsize=(3*HEIGHT, HEIGHT/2))
# plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

## multivaried series

WIN_SIZE = 10
rolling_multi = data_multi.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Appliances - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='value')
#plot_series(smooth_df_multi['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_smoothing_10.png')
show()

WIN_SIZE = 100
rolling_multi = data_multi.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Appliances - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='value')
#plot_series(smooth_df_multi['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_smoothing_100.png')
show()

