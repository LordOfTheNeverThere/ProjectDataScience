# %%

from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

index_multi = 'Date'
target_multi = 'Glucose'
data_multi = read_csv('glucose.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='consumption', title=target_multi)
plot_series(data_multi['Insulin'])
xticks(rotation = 45)
show()


# %% missing values
import pandas as pd
data = read_csv('../glucose.csv', na_values='')

data['Date'] = pd.to_datetime(data['Date'], format = "%d/%m/%Y %H:%M")
data = data.set_index('Date')


mv = {}
for var in data_multi:
    nr = data_multi[var].isna().sum()
    if nr > 0:
        mv[var] = nr
print(mv)

data.head()

def missing_values_handle(dataframe):
    hours = dataframe.index.hour.unique()
    df1 = dataframe.copy()
    medians = []

    for x in hours:
        medians.append(df1[df1.index.hour == x]['Insulin'].mean())
    
    df1['Insulin'] = df1['Insulin'].fillna(-1)

    for x in range(len(hours)):
        if sum(df1['Insulin'] == -1) != 0:
            df1.loc[(df1.index.hour == hours[x]) & (df1['Insulin'] == -1), 'Insulin'] = medians[x]
            

    return df1

data = missing_values_handle(data)

data.to_csv('mv_glocose.csv')


# %% scaling
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= ['Insulin', 'Glucose'])
tmp.to_csv('data/glucose_scaled_zscore.csv', index=False)

# %% 
from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

index_multi = 'date'
target_multi = 'Appliances'
data_multi = read_csv('data/time_series/appliances.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='consumption', title=target_multi)
plot_series(data_multi['lights'])
xticks(rotation = 45)

# %% smoothing

WIN_SIZE = 100
rolling_multi = data_multi.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Appliances - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='consumption')
plot_series(smooth_df_multi['lights'])
xticks(rotation = 45)

show()


WIN_SIZE = 10
rolling_multi = data_multi.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Appliances - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='consumption')
plot_series(smooth_df_multi['lights'])
xticks(rotation = 45)

show()

# %% aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df



figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'D')
plot_series(agg_multi_df[target_multi], title='Appliances - Daily consumptions', x_label='timestamp', y_label='consumption')
plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'W')
plot_series(agg_multi_df[target_multi], title='Appliances - Weekly consumptions', x_label='timestamp', y_label='consumption')
plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'M')
plot_series(agg_multi_df[target_multi], title='Appliances - Monthly consumptions', x_label='timestamp', y_label='consumption')
plot_series(agg_multi_df['lights'], x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
show()


# %% differenciation

diff_df_multi = data_multi.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df_multi[target_multi], title='Appliances - Differentiation', x_label=index_multi, y_label='consumption')
plot_series(diff_df_multi['lights'])
xticks(rotation = 45)
show()