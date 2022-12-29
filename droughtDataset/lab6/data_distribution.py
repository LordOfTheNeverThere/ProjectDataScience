from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

##### 5-number summary
## hour and weekly data

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
savefig('images/profiling/set2_distribution_summary.png')
show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
savefig('images/profiling/set2_distribution_boxplots.png')
## falta adicionar os títulos
show()