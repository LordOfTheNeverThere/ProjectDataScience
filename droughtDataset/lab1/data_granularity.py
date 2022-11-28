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

rows = len(variables)
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('images/granularity_study.png')
# show()
