# %%
from pandas import read_csv, to_datetime
from matplotlib.pyplot import figure, xticks, show, title, tight_layout, savefig
from ts_functions import plot_series, HEIGHT

data = read_csv('glucose.csv')


data['Date'] = to_datetime(data['Date'], format = "%d/%m/%Y %H:%M")
data = data.set_index('Date')

print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='date', y_label='value', title='Glucose')
xticks(rotation = 45)

# %% boxplots - for the most atomic series

data.boxplot(rot=90)
title("Global Boxplot")
tight_layout()
savefig('images/profiling/boxplots.png')
# %%  hourly, weekly, monthly


from matplotlib.pyplot import subplots

index = data.index.to_period('W') 
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)

index = data.index.to_period('M') 
month_df = data.copy().groupby(index).sum()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
axs[2].grid(False)
axs[2].set_axis_off()
axs[2].set_title('MONTHLY', fontweight="bold")
axs[2].text(0, 0, str(month_df.describe()))
show()

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0]) #first
week_df.boxplot(ax=axs[1]) #second
month_df.boxplot(ax=axs[2]) #second
savefig('images/profiling/boxplots_h_w_m.png')
# %% histograms - for each of granularity ; this example is hourly - update titles and legend
import matplotlib.pyplot as plt
bins = (10, 25, 50)
fig, axs = plt.subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly glucose and insulin %d bins'%bins[j])
    axs[j].set_xlabel('value')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j], label=['insulin', 'glucose'])
    axs[j].legend()

savefig('images/profiling/histograms.png')
# %%
