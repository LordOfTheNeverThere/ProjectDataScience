import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
import numpy as np

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../Data/drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
del data['fips']

# 2002/10/14 -> 2002*10000 + 10 = 200210
#ENCODING DATAS CÓDIGO YYYYMMDD

index = []
for date in data["date"]:
    index.append((date.year -2000 )* 10000 + date.month*100 + date.day)

data["date"] = index

data.to_csv('../Data/VarEncoding/datesyyyymmdd.csv', index=False)
#
#
#EPOCH TIME MUAHAHAHA
data = pd.read_csv('../Data/drought.csv', na_values='?')
del data['fips']
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
data['date'] = pd.to_datetime(data["date"]).values.astype(np.int64) // 10 ** 6
data.to_csv('../Data/VarEncoding/datesEPOCH.csv', index=False)


#CYCLICAL TYPE BEAT
data = pd.read_csv('../Data/drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')
data.insert(2, "ymonth", data['date'].dt.month, True)
data.insert(2, "xmonth", data['date'].dt.month, True)
data["xmonth"] = np.cos(2* np.pi * data["xmonth"] / 12) # x=cos(ang)
data["ymonth"] = np.sin(2* np.pi * data["ymonth"] / 12) # y=sin(ang)
data['date'] = pd.to_datetime(data["date"]).values.astype(np.int64) // 10 ** 6
del data['fips']
data.to_csv('../Data/VarEncoding/datesCyclical.csv', index=False)

print(data)

