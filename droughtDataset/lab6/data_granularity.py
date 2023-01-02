from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

## timestamp -> date
## everything is done for day, week, month, quarter, year

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])

#### granularity: day
day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Granularity: Day', x_label='timestamp', y_label='values')
xticks(rotation = 45)
## falta legenda
savefig('images/profiling/set2_granularity_day.png')
show()

#### granularity: week
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Granularity: Week', x_label='timestamp', y_label='values')
xticks(rotation = 45)
## falta legenda
savefig('images/profiling/set2_granularity_week.png')
show()

#### granularity: month
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Granularity: Month', x_label='timestamp', y_label='values')
xticks(rotation = 45)
## falta legenda
savefig('images/profiling/set2_granularity_month.png')
show()

#### granularity: quarter
index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(quarter_df, title='Granularity: Quarter', x_label='timestamp', y_label='values')
xticks(rotation = 45)
## falta legenda
savefig('images/profiling/set2_granularity_quarter.png')
show()

#### granularity: year
index = data.index.to_period('Y')
year_df = data.copy().groupby(index).mean()
year_df['timestamp'] = index.drop_duplicates().to_timestamp()
year_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(year_df, title='Granularity: Year', x_label='timestamp', y_label='values')
xticks(rotation = 45)
## falta legenda
savefig('images/profiling/set2_granularity_year.png')
show()