from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
from ds_charts import get_variable_types

#read the csv

import os
os.chdir('../Data/VarEncoding')
list = os.listdir()
#list=[]
list.append('../drought.csv')

for path in list:
    file=os.path.splitext(path)[0] 
    print(file)

    register_matplotlib_converters()
    data = read_csv(f'{file}.csv', na_values='?')
    data.describe()

    if(file == "../drought"):
        del data["fips"]

    # Determinate Outliers

    OUTLIER_PARAM: int = 1.5 # define the number of stdev to use or the IQR scale (usually 1.5)
    OPTION = 'stdev'  # or 'stdev'

    def determine_outlier_thresholds(summary5: DataFrame, var: str):
        if 'iqr' == OPTION:
            iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
            top_threshold = summary5[var]['75%']  + iqr
            bottom_threshold = summary5[var]['25%']  - iqr
        else:  # OPTION == 'stdev'
            std = OUTLIER_PARAM * summary5[var]['std']
            top_threshold = summary5[var]['mean'] + std
            bottom_threshold = summary5[var]['mean'] - std
        return top_threshold, bottom_threshold

    numeric_vars = get_variable_types(data)['Numeric']

    #Dropping Outliers

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    print('Original data:', data.shape)

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)

    df.to_csv(f'../OutliersTreat/{file}_{OPTION}_drop_outliers.csv', index=False)
    print('data after dropping outliers:', df.shape)

    #Replacing outliers with fixed value (in this case, median value)

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        median = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    print('data after replacing outliers:', df.describe())
    df.to_csv(f'../OutliersTreat/{file}_{OPTION}_replacing_outliers_median.csv', index=False)

    #Replacing outliers with fixed value (in this case, mean value)

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        median = df[var].mean()
        df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    print('data after replacing outliers:', df.describe())
    df.to_csv(f'../OutliersTreat/{file}_{OPTION}_replacing_outliers_mean.csv', index=False)

    # Truncating outliers

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

    print('data after truncating outliers:', df.describe())
    df.to_csv(f'../OutliersTreat/{file}_{OPTION}_truncate_outliers.csv', index=False)