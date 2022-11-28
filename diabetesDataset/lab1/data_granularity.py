from pandas import read_csv

filename = 'diabetesDataset/diabetic_data.csv'
data = read_csv(filename, na_values='?')
data = data.drop(['encounter_id'], axis=1)
data = data.drop(['patient_nbr'], axis=1)
data_num = data.drop(['admission_type_id'], axis=1)
data_num = data_num.drop(['discharge_disposition_id'], axis=1)
data_num = data_num.drop(['admission_source_id'], axis=1)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}



from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

variables = get_variable_types(data_num)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=100)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('diabetesDataset/lab1/images/granularity/granularity_single.png')
show()

import sys
sys.path.insert(1, '/ds_charts')
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

variables = get_variable_types(data_num)['Numeric']
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
savefig('diabetesDataset/lab1/images/granularity/granularity_study.png', dpi=400)
show()