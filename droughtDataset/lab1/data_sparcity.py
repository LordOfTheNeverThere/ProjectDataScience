#%%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('droughtDataset/drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
data.shape

def get_variable_types(data) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in data.columns[1:]:
        uniques = data[c].dropna(inplace=False).unique()

        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            data[c].astype('bool')
        elif data[c].dtype == 'datetime64[ns]':
            variable_types['Date'].append(c)
        elif data[c].dtype == 'int64' or data[c].dtype == 'float64':
            variable_types['Numeric'].append(c)
        else:
            data[c].astype('category')
            variable_types['Symbolic'].append(c)


#FIPS is a county code, so its symbolic
    data["fips"].astype('category')
    variable_types['Symbolic'].append(data["fips"])
    return variable_types

numeric_vars = list(data.columns)[20:] 
numeric_vars = [list(data.columns)[0]]+ numeric_vars

if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    print(float(i/len(numeric_vars))*100)
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):

        var2 = numeric_vars[j]
        # axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'droughtDataset/lab1/images/sparsity_study_numeric_soil.png')

