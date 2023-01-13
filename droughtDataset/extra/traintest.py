from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from ds_charts import plot_evaluation_results, bar_chart
from numpy import ndarray
import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

file_tag = 'drought_select'
data: DataFrame = read_csv(f'data_featureselect.csv')
target = 'class'
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()

# hold-out split

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'../Data/FeatureSelection/{file_tag}_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'../Data/FeatureSelection/{file_tag}_test.csv', index=False)
