# %% Imports

from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np

# %% Load Datasets

trainSet: DataFrame = read_csv('minMaxedTrainData.csv')
valSet: DataFrame = read_csv('minMaxedValData.csv')
testSet: DataFrame = read_csv('minMaxedTestData.csv')

yTrain, yVal, yTest = trainSet['readmitted'], valSet['readmitted'], testSet['readmitted']
xTrain, xVal, xTest = trainSet.drop(columns=['readmitted']), valSet.drop(columns=['readmitted']), testSet.drop(columns=['readmitted'])


labels = unique(yTrain)
labels.sort()
# %% GridSearch

evalMetric = accuracy_score
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = [1, 2, 10, 100, 1000]
values = {}
best = (0, '')
lastBest = 0
for d in dist:
    yValValues = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, p=d)
        knn.fit(xTrain, yTrain)
        predictedYVal = knn.predict(xVal)
        yValValues.append(evalMetric(yVal, predictedYVal))
        if yValValues[-1] > lastBest:
            best = (n, d)
            lastBest = yValValues[-1]
    values[d] = yValValues

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig('images/balanced1_knn_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))




# %% Get The Best Model's metrics

clf = knn = KNeighborsClassifier(n_neighbors=best[0], p=best[1])
clf.fit(xTrain, yTrain)
predictedYTrain = clf.predict(xTrain)
predictedYTest = clf.predict(xTest)
plot_evaluation_results(labels, yTrain, predictedYTrain, yTest, predictedYTest)
savefig('images/balanced1_knn_best.png')
show()
# %%

from matplotlib.pyplot import figure, savefig

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig('images/overfitting_study_knn.png')

d = 'euclidean'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
nvalues = [5, 7, 9, 11, 13, 15, 17]
for n in nvalues:
    print(n)
    knn = KNeighborsClassifier(n_neighbors=n, p=10)
    knn.fit(xTrain, yTrain)
    prd_tst_Y = knn.predict(xTest)
    prd_trn_Y = knn.predict(xTrain)
    y_tst_values.append(eval_metric(yTest, prd_tst_Y))
    y_trn_values.append(eval_metric(yTrain, prd_trn_Y))
plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))

# %%
