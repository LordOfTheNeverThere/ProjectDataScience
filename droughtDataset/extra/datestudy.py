#FAZER OS TRAINING SETS

import numpy as np
from pandas import read_csv, concat, unique, DataFrame, to_datetime
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, xticks, show, plot
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import figure, savefig, show


import os

data = read_csv('../Data/drought_prepared.csv', usecols=["date", "class"])
data_grouped = data.groupby("date").mean()

data_grouped.to_csv("date_study_prepared.csv")

figure(figsize=(3*HEIGHT, HEIGHT))
# , x_label='date', y_label='Probability of Drought', title='Probability of Drought based on the Date (Prepared Dataset)'
data.plot(x = "date", y = "class", kind = "line")
## falta legenda
savefig('images/date_study_prepared.png')
show()

data = read_csv('../Data/drought.csv', usecols=["date", "class"], sep=',', decimal='.')
data['date'] = to_datetime(data['date'], format = '%d/%m/%Y')
data_grouped = data.resample('D', on='date').mean()

data_grouped.to_csv("date_study.csv")
data_grouped= data_grouped.dropna()

print(data_grouped)
figure(figsize=(3*HEIGHT, HEIGHT))

plot_series(data_grouped["class"], x_label='date', y_label='Probability of Drought', title='Probability of Drought based on the Date (Original Dataset)')
## falta legenda
savefig('images/date_study.png')
show()

