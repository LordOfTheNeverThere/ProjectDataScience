#%%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../drought.csv', na_values='-1')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
data.shape


def get_variable_types(data) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in data.columns:
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
    # data["fips"].astype('category')
    # variable_types['Symbolic'].append(data["fips"])
    return variable_types



variable_types = get_variable_types(data)
summary5 = data[variable_types['Numeric']].describe()
# print(summary5)
# data = data[variable_types['Numeric']]




#GLOBAL BOXPLOT

from matplotlib.pyplot import savefig, show

data.boxplot(rot=45)
savefig('images/global_boxplot.png')
show()
#

#SINGLE BOXPLOTS
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

numeric_vars = get_variable_types(data)['Binary']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')
rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/single_boxplots_binary.png')
show()


#OUTLIERS
from matplotlib.pyplot import figure, savefig, show
from ds_charts import get_variable_types, multiple_bar_chart, HEIGHT

NR_STDEV: int = 2

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
        data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
savefig('images/outliers.png')
show()


HISTOGRAMS

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

numeric_vars = get_variable_types(data)['Binary']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/single_histograms_numeric_binary.png')
show()


# #DISTRIBUTION SEABORN
# from matplotlib.pyplot import savefig, show, subplots
# from seaborn import distplot
# from ds_charts import HEIGHT, get_variable_types
#
# numeric_vars = get_variable_types(data)['Numeric']
# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')
#
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(numeric_vars)):
#     axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
#     distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# savefig('images/histograms_trend_numeric.png')
# show()
#
#
#DISTRIBUTION REAL SHIT LETS GO
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm
from matplotlib.pyplot import savefig, show, subplots, Axes
from ds_charts import HEIGHT, multiple_line_chart, get_variable_types
from ds_charts import get_variable_types, choose_grid, HEIGHT
numeric_vars = get_variable_types(data)['Numeric']
rows, cols = choose_grid(len(numeric_vars))
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
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var)

if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

#with the changes

i, j = 0, 0
for n in range(4, 19):
    print(n)
    save = "images/dist/histogram_numeric_distribution" + str(n) + ".png"
    fig, axs = subplots(1, 1, figsize=(1*HEIGHT, 1*HEIGHT), squeeze=Falsgite)
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    savefig(save)
# show()


