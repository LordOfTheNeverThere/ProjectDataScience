from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
index_multi = 'date'
target_multi = 'QV2M'
data_multi = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

#### aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

## multivaried series

granularity = ('D', 'W', 'M', 'Q', 'Y')
for j in range(len(granularity)):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_multi_df = aggregate_by(data_multi, index_multi, granularity[j])
    plot_series(agg_multi_df[target_multi], title=f'{target_multi} - {granularity[j]} values', x_label='timestamp', y_label='value')
    #plot_series(agg_multi_df['lights'])
    xticks(rotation = 45)
    savefig(f'images/transformation/set2_data_aggregation_{granularity[j]}.png')
    show()