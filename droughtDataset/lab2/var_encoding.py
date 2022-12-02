import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../drought.csv', na_values='?')
data['date'] = pd.to_datetime(data['date'], format = '%d/%m/%Y')

#ENCODING DATAS MÊS a MÊS linearmente
