from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

## everything is done for day, week, month, quarter, year

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', usecols=["date", "QV2M"], sep=',', decimal='.', parse_dates=True, dayfirst=True)

#### prepare data
from matplotlib.pyplot import subplots
# day
day_df = data.copy().groupby(data.index.date).mean()
# week
index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
# month
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
# quarter
index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('timestamp', drop=True, inplace=True)
# year
index = data.index.to_period('Y')
year_df = data.copy().groupby(index).mean()
year_df['timestamp'] = index.drop_duplicates().to_timestamp()
year_df.set_index('timestamp', drop=True, inplace=True)

# ##### 5-number summary
# ## day data
# _, axs = subplots(1, 5, figsize=(4*HEIGHT, HEIGHT/2))
# axs[0].grid(False)
# axs[0].set_axis_off()
# axs[0].set_title('DAILY', fontweight="bold")
# axs[0].text(0, 0, str(day_df['QV2M'].describe()))
# ## week data
# axs[1].grid(False)
# axs[1].set_axis_off()
# axs[1].set_title('WEEKLY', fontweight="bold")
# axs[1].text(0, 0, str(week_df['QV2M'].describe()))
# ## month data
# axs[2].grid(False)
# axs[2].set_axis_off()
# axs[2].set_title('MONTHLY', fontweight="bold")
# axs[2].text(0, 0, str(month_df['QV2M'].describe()))
# ## quarter data
# axs[3].grid(False)
# axs[3].set_axis_off()
# axs[3].set_title('QUARTERLY', fontweight="bold")
# axs[3].text(0, 0, str(quarter_df['QV2M'].describe()))
# ## year data
# axs[4].grid(False)
# axs[4].set_axis_off()
# axs[4].set_title('YEARLY', fontweight="bold")
# axs[4].text(0, 0, str(year_df['QV2M'].describe()))
# savefig('images/profiling/set2_distribution_QV2M_summary.png')
## falta adicionar os títulos
#show()
#### boxplots 
_, axs = subplots(1, 5, figsize=(5*HEIGHT, HEIGHT/2))

axs[0].set_title("day")
axs[1].set_title("week")
axs[2].set_title("month")
axs[3].set_title("quarter")
axs[4].set_title("year")

data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
month_df.boxplot(ax=axs[2])
quarter_df.boxplot(ax=axs[3])
year_df.boxplot(ax=axs[4])


savefig('images/profiling/set2_distribution_QV2M_boxplots_new.png')
## falta adicionar os títulos
#show()

# ##### variables distribution
# bins = (5, 10, 25, 50)
# ## day data
# _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT/2))
# for j in range(len(bins)):
#     axs[j].set_title('%d bins daily values'%bins[j])
#     axs[j].set_xlabel('values')
#     axs[j].set_ylabel('Nr records')
#     axs[j].hist(day_df['QV2M'].values, bins=bins[j])
# savefig('images/profiling/set2_distribution_QV2M_variables_day.png')
# #show()
# ## week data
# _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT/2))
# for j in range(len(bins)):
#     axs[j].set_title('%d bins weekly values'%bins[j])
#     axs[j].set_xlabel('values')
#     axs[j].set_ylabel('Nr records')
#     axs[j].hist(week_df['QV2M'].values, bins=bins[j])
# savefig('images/profiling/set2_distribution_QV2M_variables_week.png')
# #show()
# ## month data
# _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT/2))
# for j in range(len(bins)):
#     axs[j].set_title('%d bins monthly values'%bins[j])
#     axs[j].set_xlabel('values')
#     axs[j].set_ylabel('Nr records')
#     axs[j].hist(month_df['QV2M'].values, bins=bins[j])
# savefig('images/profiling/set2_distribution_QV2M_variables_month.png')
# #show()
# ## quarter data
# _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT/2))
# for j in range(len(bins)):
#     axs[j].set_title('%d bins quarterly values'%bins[j])
#     axs[j].set_xlabel('values')
#     axs[j].set_ylabel('Nr records')
#     axs[j].hist(quarter_df['QV2M'].values, bins=bins[j])
# savefig('images/profiling/set2_distribution_QV2M_variables_quarter.png')
# #show()
# ## year data
# _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT/2))
# for j in range(len(bins)):
#     axs[j].set_title('%d bins yearly values'%bins[j])
#     axs[j].set_xlabel('values')
#     axs[j].set_ylabel('Nr records')
#     axs[j].hist(year_df['QV2M'].values, bins=bins[j])
# savefig('images/profiling/set2_distribution_QV2M_variables_year.png')
# ##show()
