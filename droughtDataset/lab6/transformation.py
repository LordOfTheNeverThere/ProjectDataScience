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

#### aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'D')
# plot_series(agg_df, title='Daily values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'W')
# plot_series(agg_df, title='Weekly values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'M')
# plot_series(agg_df, title='Monthly values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

## multivaried series

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'D')
plot_series(agg_multi_df[target_multi], title='Appliances - Daily values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_day.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'W')
plot_series(agg_multi_df[target_multi], title='Appliances - Weekly values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_week.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'M')
plot_series(agg_multi_df[target_multi], title='Appliances - Monthly values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'], x_label='timestamp', y_label='value')
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_month.png')
show()

#### differentiation

# diff_df = data.diff()
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(diff_df, title='Differentiation', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

## multivaried series

diff_df_multi = data_multi.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df_multi[target_multi], title='Appliances - Differentiation', x_label=index_multi, y_label='value')
#plot_series(diff_df_multi['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_differentiation.png')
show()