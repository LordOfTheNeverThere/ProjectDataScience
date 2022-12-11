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

data: DataFrame = read_csv('zScoredData.csv')
y = data['readmitted']
X =  data.drop(columns=['readmitted', 'Unnamed: 0'])


xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, train_size=0.6, random_state=42)
xTrain, xVal, yTrain, yVal = sklearn.model_selection.train_test_split(
    xTrain, yTrain, train_size=0.5, random_state=42)

labels = unique(yTrain)
labels.sort()
# %% GridSearch

evalMetric = accuracy_score
nvalues = np.arange(175,1001,25)
dist = np.arange(50, 1001, 50)  # Add more distance metrics
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
savefig('images/{file_tag}_knn_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))




# %% Get The Best Model's metrics

clf = knn = KNeighborsClassifier(n_neighbors=best[0], p=best[1])
clf.fit(xTrain, yTrain)
predictedYTrain = clf.predict(xTrain)
predictedYTest = clf.predict(xTest)
plot_evaluation_results(labels, yTrain, predictedYTrain, yTest, predictedYTest)
savefig('images/{file_tag}_knn_best.png')
show()
# %%
