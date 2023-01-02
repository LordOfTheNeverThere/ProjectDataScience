from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
index_multi = 'date'
target_multi = 'QV2M'
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

#### original

# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(data, x_label='timestamp', y_label='value', title='Drought Forecasting Original')
# xticks(rotation = 45)
# savefig('images/transformation/set2_data_transformation_original.png')
# show()

#### QV2M

figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='value', title=target_multi)
#plot_series(data_multi['lights'])
xticks(rotation = 45)
savefig(f'images/transformation/set2_data_transformation_{target_multi}.png')
show()

#### smoothing multivaried series

# WIN_SIZE = (10, 50, 100)
# for j in range(len(WIN_SIZE)):
#     rolling_multi = data_multi.rolling(window=WIN_SIZE[j])
#     smooth_df_multi = rolling_multi.mean()
#     figure(figsize=(3*HEIGHT, HEIGHT))
#     plot_series(smooth_df_multi[target_multi], title=f'{target_multi} - Smoothing (win_size={WIN_SIZE[j]})', x_label=index_multi, y_label='value')
#     #plot_series(smooth_df_multi['lights'])
#     xticks(rotation = 45)
#     savefig(f'images/transformation/set2_data_smoothing_{target_multi}_{WIN_SIZE[j]}.png')
#     show()