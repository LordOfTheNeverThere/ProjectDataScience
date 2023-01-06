#%% Imports
from pandas import Series
from numpy import ones
from pandas import read_csv
import matplotlib
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
import pandas as pd

#%% Hourly Distro

data = read_csv('glucose.csv')
data['Date'] = pd.to_datetime(data['Date'], format = "%d/%m/%Y %H:%M")
data = data.set_index('Date') ## Droping Index col
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(data['Insulin'], x_label='timestamp', y_label='Insulin', title='Insulin Hourly Data')
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(data['Glucose'], x_label='timestamp', y_label='Glucose', title='Glucose Hourly Data')
xticks(rotation = 45)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(data['Insulin'], x_label='timestamp',
            y_label='Insulin', title='Insulin Hourly Data')
plot_series(data['Glucose'], x_label='timestamp',
            y_label='Glucose', title='Glucose Hourly Data')
xticks(rotation=45)
show()


# %% Daily Distro
day_df = data.copy().groupby(data.index.date).mean()
day_df.to_csv('dailyGlucose.csv')


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(day_df['Insulin'], x_label='timestamp', y_label='Insulin', title='Insulin Daily Data')
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(day_df['Glucose'], x_label='timestamp', y_label='Glucose', title='Glucose Daily Data' )
xticks(rotation = 45)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(day_df['Insulin'], x_label='timestamp',
            y_label='Insulin', title='Insulin Daily Data')
plot_series(day_df['Glucose'], x_label='timestamp',
            y_label='Glucose', title='Glucose Daily Data')
xticks(rotation=45)
show()
# %% Weekly Distro

index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['Date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('Date', drop=True, inplace=True)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(week_df['Insulin'], x_label='timestamp', y_label='Insulin', title='Insulin Weekly Data')
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(week_df['Glucose'], x_label='timestamp', y_label='Glucose', title='Glucose Weekly Data' )
xticks(rotation = 45)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(week_df['Insulin'], x_label='timestamp',
            y_label='Insulin', title='Insulin Weekly Data')
plot_series(week_df['Glucose'], x_label='timestamp',
            y_label='Glucose', title='Glucose Weekly Data')
xticks(rotation=45)
show()
# %% Montly Distro

index = data.index.to_period('M')
monthly = data.copy().groupby(index).mean()
monthly['Date'] = index.drop_duplicates().to_timestamp()
monthly.set_index('Date', drop=True, inplace=True)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(monthly['Insulin'], x_label='timestamp', y_label='Insulin', title='Insulin Monthly Data')
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(monthly['Glucose'], x_label='timestamp', y_label='Glucose', title='Glucose Monthly Data' )
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(monthly['Insulin'], x_label='timestamp',
            y_label='Insulin', title='Insulin Monthly Data')
plot_series(monthly['Glucose'], x_label='timestamp',
            y_label='Glucose', title='Glucose Monthly Data')
xticks(rotation=45)


show()
# %% Quarterly Distro

index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['Date'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('Date', drop=True, inplace=True)


figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(quarter_df['Insulin'], x_label='timestamp', y_label='Insulin', title='Insulin Quarterly Data')
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(quarter_df['Glucose'], x_label='timestamp', y_label='Glucose', title='Glucose Quarterly Data' )
xticks(rotation = 45)

figure(figsize=(5*HEIGHT, HEIGHT))
plot_series(quarter_df['Insulin'], x_label='timestamp',
            y_label='Insulin', title='Insulin Quarterly Data')
plot_series(quarter_df['Glucose'], x_label='timestamp',
            y_label='Glucose', title='Glucose Quarterly Data')
xticks(rotation=45)

show()
# %%
## Interesting granularities are hourly, weekly, monthly

# %% Stationary Studies Glucose


dt_series = Series(data['Glucose'])

mean_line = Series(ones(len(dt_series.values)) *
                   dt_series.mean(), index=dt_series.index)
series = {'Glucose': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='timestamp', y_label='consumption',
            title='Stationary study', show_std=True)
show()


BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'Glucose': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='time', y_label='consumptions',
            title='Stationary study', show_std=True)
show()

# %% '' Insulin


dt_series = Series(data['Insulin'])

mean_line = Series(ones(len(dt_series.values)) *
                   dt_series.mean(), index=dt_series.index)
series = {'Insulin': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='timestamp', y_label='consumption',
            title='Stationary study', show_std=True)
show()


BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'Insulin': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='time', y_label='consumptions',
            title='Stationary study', show_std=True)
show()

# %%
