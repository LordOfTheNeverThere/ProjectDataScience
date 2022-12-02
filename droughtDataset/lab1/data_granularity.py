import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

# variables = get_variable_types(data)['Numeric']
# if [] == variables:
#     raise ValueError('There are no numeric variables.')
#
# rows, cols = choose_grid(len(variables))
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(variables)):
#     axs[i, j].set_title('Histogram for %s'%variables[n])
#     axs[i, j].set_xlabel(variables[n])
#     axs[i, j].set_ylabel('nr records')
#     axs[i, j].hist(data[variables[n]].values, bins=100)
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# savefig('images/granularity_single.png')
# show()

#GRANULARITY PARA TODOS ISTO VAI CORRER MAL
variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

print(len(data["fips"].unique()))

rows = len(variables)+1
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows-1):
    if(i == 0 or i > 18):
        bins = (10, 25, 50)
    else:
        bins = (10, 100, 1000)
    for j in range(cols):


        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
#
savefig('images/granularity_study.png')

#ANALISAR AS DATAS
rows = 1
cols = 3
fig, axs = subplots(1, 1, figsize=(1*HEIGHT, 1*HEIGHT), squeeze=False)
i = 0
j = 0
bins = (10, 100, 1000)
year = data["date"].groupby(data["date"].dt.year).count()
year = year.to_dict()

data["count"] = 1
new_data = data[["date", "count"]]
# new_data.plot.hist(column = "date", y = "count", rot = 0)
# savefig('images/granularity_dates_study3.png')

# new_data.groupby(pd.Grouper(key = "date", freq='M'))
new_data = new_data.resample(rule='M', on='date')['count'].sum()
print(new_data)
# bar(list(month.keys()), list(month.values()))
month = new_data.to_dict()



j=0
fig, axs = subplots(1, 1, figsize=(1*HEIGHT, 1*HEIGHT), squeeze=False)

axs[i, j].set_title('Histogram for dates by Year')
axs[i, j].set_xlabel("date")
axs[i, j].set_ylabel('Nr records')
axs[i, j].bar(list(year.keys()), list(year.values()))
savefig('images/granularity_dates_study1.png')

j=0
fig, axs = subplots(1, 1, figsize=(1*HEIGHT, 1*HEIGHT), squeeze=False)

axs[i, j].set_title('Histogram for dates by Month')
axs[i, j].set_xlabel("date")
axs[i, j].set_ylabel('Nr records')
new_data.plot.bar(x = "date", y = "count", rot = 0)

savefig('images/granularity_dates_study2.png')

# show()
