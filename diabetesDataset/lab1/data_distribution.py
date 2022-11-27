from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
filename = 'diabetesDataset/diabetic_data.csv'
data = read_csv(filename, na_values='?')
data = data.drop(['encounter_id'], axis=1)
data = data.drop(['patient_nbr'], axis=1)
data_num = data.drop(['admission_type_id'], axis=1)
data_num = data_num.drop(['discharge_disposition_id'], axis=1)
data_num = data_num.drop(['admission_source_id'], axis=1)
summary5 = data.describe()
summary5

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

from matplotlib.pyplot import savefig, show, title, tight_layout
from ds_charts import get_variable_types, choose_grid, HEIGHT

# data_num.boxplot(rot=90)
# title("Global Boxplot")
# tight_layout()
# savefig('diabetesDataset/lab1/images/global_boxplot.png')

# from matplotlib.pyplot import savefig, show, subplots
# import sys
# sys.path.insert(1, '/ds_charts')


# numeric_vars = get_variable_types(data_num)['Numeric']  # confirmar que temos q mudar para int64 e float64 e category to object
# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')
# rows, cols = choose_grid(len(numeric_vars))
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(numeric_vars)):
#     axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
#     axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# savefig('diabetesDataset/lab1/images/single_boxplots.png')
# show()


# from matplotlib.pyplot import figure, savefig, show
# from ds_charts import get_variable_types, multiple_bar_chart, HEIGHT

# NR_STDEV: int = 2

# numeric_vars = get_variable_types(data_num)['Numeric']
# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')

# outliers_iqr = []
# outliers_stdev = []
# summary5 = data.describe(include='number')

# for var in numeric_vars:
#     iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
#     outliers_iqr += [
#         data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
#         data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
#     std = NR_STDEV * summary5[var]['std']
#     outliers_stdev += [
#         data[data[var] > summary5[var]['mean'] + std].count()[var] +
#         data[data[var] < summary5[var]['mean'] - std].count()[var]]

# outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
# figure(figsize=(12, HEIGHT))
# multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
# tight_layout()
# savefig('diabetesDataset/lab1/images/outliers.png')
# show()


# from matplotlib.pyplot import savefig, show, subplots
# from ds_charts import get_variable_types, choose_grid, HEIGHT
# from collections import Counter
# import numpy
# numeric_vars = get_variable_types(data_num)['Numeric']
# # permite contar o nr de vezes que aparece cada valor
# # recounted = Counter(data[numeric_vars[8]])
# # print(recounted)

# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')

# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(numeric_vars)):
#     axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
#     axs[i, j].set_xlabel(numeric_vars[n])
#     axs[i, j].set_ylabel("nr records")
#     axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# # permite ver quais os bins e quantos valores
# # print(numpy.histogram(data[numeric_vars[9]].dropna().values, 'auto'))
# savefig('diabetesDataset/lab1/images/single_histograms_numeric.png')
# show()



# from matplotlib.pyplot import savefig, show, subplots
# from seaborn import distplot
# from ds_charts import HEIGHT, get_variable_types

# numeric_vars = get_variable_types(data_num)['Numeric']
# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')

# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(numeric_vars)):
#     axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
#     distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# savefig('diabetesDataset/lab1/images/histograms_trend_numeric.png')
# show()

# falta confirmar estes

from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm
from matplotlib.pyplot import savefig, show, subplots, Axes
from ds_charts import HEIGHT, multiple_line_chart, get_variable_types

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

numeric_vars = get_variable_types(data_num)['Numeric']
rows, cols = choose_grid(len(numeric_vars))
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('diabetesDataset/lab1/images/histogram_numeric_distribution.png')
show()

# falta correr a partir daqui


########### perceber o que fazer com diag1, diag2, etc pos6-9::::::::::: o mighel ai fazer o encoding e depois é so usar aqui


# from matplotlib.pyplot import savefig, show, subplots
# from ds_charts import HEIGHT, choose_grid, get_variable_types, bar_chart

# symbolic_vars = get_variable_types(data)['Symbolic']
# symbolic_vars = symbolic_vars[6:9]
# if [] == symbolic_vars:
#     raise ValueError('There are no symbolic variables.')

# rows, cols = choose_grid(len(symbolic_vars))
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(symbolic_vars)):
    
#     counts = data[symbolic_vars[n]].value_counts()

#     x_val = counts.index.to_list()
#     y_val =  counts.values

#     if symbolic_vars[n] in ['age']:
#         x_val = [counts.index.to_list()[i] for i in [9, 8, 7, 5, 4, 2, 1, 0, 3, 6]]
#         y_val = [counts.values[i] for i in [9, 8, 7, 5, 4, 2, 1, 0, 3, 6]]

#     if symbolic_vars[n] in ['weight']:
#         x_val = [counts.index.to_list()[i] for i in [5, 4, 1, 0, 2, 3, 6, 7, 8]]
#         y_val = [counts.values[i] for i in [5, 4, 1, 0, 2, 3, 6, 7, 8]]

#     bar_chart(x_val, y_val, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation = 45)
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# fig.subplots_adjust(hspace=.9)
# savefig('diabetesDataset/lab1/images/histograms_symbolic.png')
# show()