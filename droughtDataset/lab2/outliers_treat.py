from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

#read the csv

import os
os.chdir('../Data/VarEncoding')
list = os.listdir()
#list=[]
# list.append('../drought.csv')

# for path in list:
# file=os.path.splitext(path)[0]

#aqui estamos só a escolher um dos encodings para as datas
file = "datesCyclical"
print(file)

register_matplotlib_converters()
data = read_csv(f'{file}.csv', na_values='?')
data.describe()
variable_types = get_variable_types(data)
print(variable_types['Numeric'])
# if(file == "../drought"):
#     del data["fips"]
numeric_vars = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'lat', 'lon', 'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']


#tiramos tudo até ao TS para outliers, porque os outros pareciam valores contínuos.
numeric_vars_stdev = ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'lat', 'lon', 'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']

# Determinate Outliers

#FOR LATER, ESCOLHEMOS 5 SIGMA PORQUE PARECE SER UM BOM COMPROMISO PARA AS VARIAVEIS QUE PARECEM MAIS CONTÍNUAS

OUTLIER_PARAM: int = 5 # define the number of stdev to use or the IQR scale (usually 1.5)
options = ['stdev']  # or 'stdev'
for OPTION in options:
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

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'../../lab2/images/OutliersTreat/{file}_{OPTION}_drop_outliers.png')

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

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'../../lab2/images/OutliersTreat/{file}_{OPTION}_replacing_outliers_median.png')





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



    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'../../lab2/images/OutliersTreat/{file}_{OPTION}_replacing_outliers_mean.png')






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


    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(df[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'../../lab2/images/OutliersTreat/{file}_{OPTION}_truncate_outliers.png')
