# separate the target variable from the rest of the data

import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

# file_tag = 'drought_prepared'
# data: DataFrame = read_csv(f'../Data/{file_tag}.csv')
# target = 'class'
# positive = 1
# negative = 0
# values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

# y: np.ndarray = data.pop(target).values
# X: np.ndarray = data.values
# labels: np.ndarray = unique(y)
# labels.sort()

# hold-out split

# trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

# train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
# train.to_csv(f'../Data/TrainTest/{file_tag}_train.csv', index=False)

# test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
# test.to_csv(f'../Data/TrainTest/{file_tag}_test.csv', index=False)
# values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
# values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

# plt.figure(figsize=(12,4))
# ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
# plt.show()

# NB

file_tag = 'drought_scaled'
filename = 'drought_scaled'
target = 'class'

train: DataFrame = read_csv(f'../Data/Balancing/{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'../Data/Balancing/{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig('images/drought_balanced_nb_best.png')
show()

# accuracy

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(trnX, trnY)
clf.score(tstX, tstY)

# confusion matrix

from numpy import ndarray
from sklearn.metrics import confusion_matrix

labels: ndarray = unique(y)
labels.sort()
prdY: ndarray = clf.predict(tstX)
cnf_mtx_tst: ndarray = confusion_matrix(tstY, prdY, labels=labels)
cnf_mtx_tst

# accuracy of different parameterisations

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_nb_study.png')
show()