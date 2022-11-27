#%%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../diabetic_data.csv', na_values='?')

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



def get_variable_types(df) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int64' or df[c].dtype == 'float64':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)
    return variable_types

variable_types = get_variable_types(data)

counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Frequency in var type')
savefig('images/variable_types.png')
show()
# %%
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Number of Missing values per var',
             ylabel='Number of Missing values', rotation=True)
savefig('images/mv.png')
show()
# %% Correlation Analysis
corr_mtx = abs(data.corr())
print(corr_mtx)

# %%
from matplotlib.pyplot import figure, savefig, show, title
from seaborn import heatmap

fig = figure(figsize=[12, 12])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Reds')
title('Correlation analysis')
savefig(f'images/correlation_analysis.png')
show()
