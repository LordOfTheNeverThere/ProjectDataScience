#%%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
data.shape
# %% #


figure(figsize=(4,2))
values = {'num Of Records': data.shape[0], 'num Of Variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Comparison between number of vars and dims')
savefig('images/records_variables.png')
show()
# %%
data.dtypes
# %%



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



variable_types = get_variable_types(data)

counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Frequency in var type')
savefig('images/variable_types.png')
show()

print(variable_types['Numeric'])
# %%MISSING VALUES
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
    else:
        mv[var] = 0
figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/mv.png')
show()



#%% Correlation Analysis
data = data[variable_types['Numeric']]
corr_mtx = abs(data.corr())
print(corr_mtx)

# %%
from matplotlib.pyplot import figure, savefig, show, title
from seaborn import heatmap

fig = figure(figsize=[47, 47])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Reds')
title('Correlation analysis')
savefig(f'images/correlation_analysis.png')
show()
# %%
