# %% Imports
from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

# %% Get Glucose Data read_csv
data = Dataframe()
# %% Differentiation  The role of this technique is to adress non stationary data, removing the influence of the trend I think
data = data.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data['Glucose'], title='Diabetes - Differentiation', y_label='Glucose')
plot_series(data['Insulin'], y_label='Insulin')
xticks(rotation=45)
show()
