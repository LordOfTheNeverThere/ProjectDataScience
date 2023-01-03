# %% Imports
from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

# %% Get Glucose Data read_csv
data = read_csv()


# %% 1st Differentiation  The role of this technique is to adress non stationary data, removing the influence of the trend I think

dataDiff1 = data.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data['Glucose'], title='Diabetes - Differentiation', y_label='Glucose')
plot_series(data['Insulin'], y_label='Insulin')
xticks(rotation=45)
show()

# %% 2nd Differentiation  The role of this technique is to adress non stationary data, removing the influence of the trend I think

dataDiff2 = dataDiff1.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data['Glucose'],
            title='Diabetes - Differentiation', y_label='Glucose')
plot_series(data['Insulin'], y_label='Insulin')
xticks(rotation=45)
show()
