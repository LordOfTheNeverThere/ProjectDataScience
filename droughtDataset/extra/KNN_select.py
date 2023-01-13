import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
from ds_charts import plot_evaluation_results, bar_chart
import ds_charts as ds
from sklearn.model_selection import train_test_split
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

file = "Feature Selection"
file_tag = 'drought'
filename = 'datesCyclical_stdev_drop_outliers_scaled_minmax'
target = 'class'


file_tag = 'drought_select'
filename = 'drought_select'
target = 'class'

train: DataFrame = read_csv('../Data/FeatureSelection/drought_select_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('../Data/FeatureSelection/drought_select_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

eval_metric = accuracy_score
# nvalues = [1, 3, 5]
# dist = ['manhattan']
nvalues = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
xvalues = []
yvalues = []
# for d in dist:
#     print(d)
#     xvalues.append(d)
#     y_tst_values = []
#     current_best = 0
#     for n in nvalues:
#         knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#         knn.fit(trnX, trnY)
#         prd_tst_Y = knn.predict(tstX)
#         y_tst_values.append(eval_metric(tstY, prd_tst_Y))
#
#         if y_tst_values[-1] > last_best:
#             best = (n, d)
#             last_best = y_tst_values[-1]
#         if y_tst_values[-1] > current_best:
#             current_best = y_tst_values[-1]
#
#     yvalues.append(current_best)
#     values[d] = y_tst_values
#
# figure()
# bar_chart(xvalues, yvalues, title=f'Comparison of KNN distances for {file}', ylabel='accuracy', percentage=True)
# savefig(f'images/KNN/{file}_KNN_dist.png')
# # show()
#
# figure()
# multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
# savefig(f'images/KNN/{file}_knn_study.png')
# # show()
# print('Best results with %d neighbors and %s'%(best[0], best[1]))
#
# clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
# clf.fit(trnX, trnY)
# prd_trn = clf.predict(trnX)
# prd_tst = clf.predict(tstX)
# plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
# savefig(f'images/KNN/{file}_knn_best.png')
# # show()

from matplotlib.pyplot import figure, savefig
#
def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'images/KNN/overfitting_{name}_{file}.png')

d = 'euclidean'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in nvalues:
    print(n)
    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    prd_tst_Y = knn.predict(tstX)
    prd_trn_Y = knn.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel='accuracy')

