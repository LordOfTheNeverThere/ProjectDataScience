# %% Imports
from numpy import std, argsort
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
import matplotlib as plt
from matplotlib.pyplot import figure, subplots, savefig, show, plot
from sklearn.ensemble import RandomForestClassifier
from ds_charts import set_elements, plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np
from ds_charts import overfittingStudyImage
import pandas as pd

# %% Load File Tag
file_tag = 'dummy'


# %% Load Datasets

data: DataFrame = read_csv('zScoredData.csv')
data = data.head(1000) # DELETE THIS
y = data['readmitted']
X = data.drop(columns=['readmitted', 'Unnamed: 0', 'Unnamed: 0.1', 'encounter_id'])


xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(
    X, y, train_size=0.7, random_state=42)
xTest, xVal, yTest, yVal = sklearn.model_selection.train_test_split(
    xTest, yTest, train_size=0.3333333333333333333, random_state=42)

labels = unique(yTrain)
labels.sort()

# %% Grid Search
n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
max_depths = [5, 10, 25]
max_features = [.25, .5, .75, 1]
best = ('', 0, 0)
last_best = 0
best_model = None


cols = len(max_depths)

figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)

overfittingRows = len(max_depths)
overfittingCols = len(max_features)

figure()
figOverfitting, axsOverfitting = subplots(overfittingRows, overfittingCols, figsize=(
    len(n_estimators)*HEIGHT, overfittingRows * HEIGHT), squeeze=False)
figOverfitting.suptitle('Overfitting Test')
eval_metric = accuracy_score


for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for maxFeatureIndex, f in enumerate(max_features):
        yvalues = []
        yTrainMetrics = []
        
        for  n in n_estimators:
            rf = RandomForestClassifier(
                n_estimators=n, max_depth=d, max_features=f)
            rf.fit(xTrain, yTrain)
            prdY = rf.predict(xVal)
            prdYTrain = rf.predict(xTrain)
            yvalues.append(eval_metric(yVal, prdY))
            yTrainMetrics.append(eval_metric(yTrain, prdYTrain))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = rf

        axis = set_elements(
            ax=axsOverfitting[k, maxFeatureIndex], title='MaxNumOfFeatures = ' + str(f) + ', MaxDepth = '+ str(k), xlabel=eval_metric, ylabel='NumOfEstimators')
        axis.plot(n_estimators, yTrainMetrics, color = 'orange', linewidth=2, markersize=12, label= 'Train')
        axis.plot(n_estimators, yvalues, color='blue', linewidth=2, markersize=12, label= 'Validation')
        axis.set_ylim((0,1))
        axis.legend(loc="lower right")
        



        values[f] = yvalues
        

        # overfittingStudyImage(n_estimators, yTrainMetrics, yvalues,
        #                       name=f'RF_depth={k}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric), fig=figOverfitting, axes=axsOverfitting[row, :])
        

    # figOverfitting.savefig(f'images/{file_tag}_rf_overFittingStudy.png')
    # show()

    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                        xlabel='nr estimators', ylabel='accuracy', percentage=True)

fig.savefig(f'images/{file_tag}_rf_study.png')
show()
print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f' %
      (best[0], best[1], best[2], last_best))

# %% Best Model params

prd_trn = best_model.predict(xTrain)
prd_tst = best_model.predict(xTest)
plot_evaluation_results(labels, yTrain, prd_trn, yTest, prd_tst)
savefig(f'images/{file_tag}_rf_best.png')
show()

# %% Get Features Importance

train = xTrain
variables = train.columns
importances = best_model.feature_importances_
stdevs = std(
    [tree.feature_importances_ for tree in best_model.estimators_], axis=0)
indices = argsort(importances)[::-1]
elems = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, importances[indices], stdevs[indices],
                     title='Random Forest Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/{file_tag}_rf_ranking.png')

# %%