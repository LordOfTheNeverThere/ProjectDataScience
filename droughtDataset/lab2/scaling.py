from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat

#read the csv

import os
os.chdir('../Data/OutliersTreat')
list = os.listdir()
#list=[]
list.append('../drought.csv')

for path in list:
    file=os.path.splitext(path)[0] 
    print(file)

    register_matplotlib_converters()
    data = read_csv(f'{file}.csv', na_values='?')
    data.describe()

    # Separating the Dataframes according to the type of variable

    from ds_charts import get_variable_types

    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    # Standart Scaler (z-score transformation)

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_zscore.to_csv(f'../Scaled/{file}_scaled_zscore.csv', index=False)