import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
import numpy as np

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../Data/drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')


# 2002/10/14 -> 2002*10000 + 10 = 200210
#ENCODING DATAS CÓDIGO YYYYMMDD

# index = []
# for date in data["date"]:
#     index.append((date.year -2000 )* 10000 + date.month*100 + date.day)
#
# data["date"] = index
#
# data.to_csv('../Data/datesyyyymmdd.csv')
#
#
# #EPOCH TIME MUAHAHAHA
# data = pd.read_csv('../Data/drought.csv', na_values='?')
# data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
# data['date'] = pd.to_datetime(data["date"]).values.astype(np.int64) // 10 ** 6
# data.to_csv('../Data/datesEPOCH.csv')


#CYCLICAL TYPE BEAT
data = pd.read_csv('../Data/drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')



