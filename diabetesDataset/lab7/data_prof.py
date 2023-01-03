# %%
from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, title, tight_layout, savefig
from ts_functions import plot_series, HEIGHT

data = read_csv('glucose.csv', index_col='Date', sep=',', parse_dates=True, infer_datetime_format=True)
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='date', y_label='value', title='Glucose')
xticks(rotation = 45)
show()

# %% boxplots - for the most atomic series

data.boxplot(rot=90)
title("Global Boxplot")
tight_layout()
# %%
from matplotlib.pyplot import subplots

index = data.index.to_period('W') 
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0]) #first
week_df.boxplot(ax=axs[1]) #second
show()

# %% histograms - for each of granularity ; this example is hourly - update titles and legend

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly meter_reading %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
show()
# %%
