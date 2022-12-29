from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show

## timestamp -> date

data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
