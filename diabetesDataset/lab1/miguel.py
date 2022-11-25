#%%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../diabetic_data.csv', na_values='na')

data.shape
# %% #

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

figure(figsize=(4,2))
values = {'num Of Records': data.shape[0], 'num Of Variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Comparison between number of vars and dims')
savefig('images/records_variables.png')
show()
# %%
