from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import subplots, figure, savefig #show

#read the csv

import os
os.chdir('../Data/OutliersTreat')
list = os.listdir()
#list=[]
# list.append('../drought.csv')

# for path in list:
#     file=os.path.splitext(path)[0]

file = "datesCyclical_stdev_drop_outliers"
print(file)

register_matplotlib_converters()
data = read_csv(f'{file}.csv', na_values='?')
data.describe()
if(file.split('_')[0] == 'datesCyclical' or file.split('_')[0] == "datesEPOCH"):
    data["date"] = data['date'] / 10**11
elif(file.split('_')[0] == 'datesyyyymmdd'):
    data["date"] = data['date'] / 100000

# Separating the Dataframes according to the type of variable

from ds_charts import get_variable_types

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]

df_bool = (df_bool-df_bool.min())/(df_bool.max()-df_bool.min()) # transform bools into 0 or 1

# we apply scaling both to numeric and boolean variables

# Standart Scaler (z-score transformation)

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1) # booleans and symbolic aren't being transformed (z score)
norm_data_zscore.to_csv(f'../Scaled/{file}_scaled_zscore.csv', index=False)

# MinMax Scaler

#df_nr = df_nr + df_bool
#numeric_vars = numeric_vars + boolean_vars

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1) # symbolic aren't being transformed (z score)
norm_data_minmax.to_csv(f'../Scaled/{file}_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())

# Show Results

fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score Normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax Normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
savefig(f'../../lab2/images/scaling_boxplots/{file}_boxplot.png')
#show()

# Save
#norm_data_zscore.to_csv('data/algae_scaled_zscore.csv', index=False)



