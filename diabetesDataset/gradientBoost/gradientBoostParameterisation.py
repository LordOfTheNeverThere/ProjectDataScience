# %% Imports
from numpy import std, argsort
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
import matplotlib as plt
from matplotlib.pyplot import figure, subplots, savefig, show, tight_layout
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import set_elements, plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np
import pandas as pd

# %% Load File Tag
file_tag = 'v1.0_min_samples=True'


# %% Load Datasets

trainSet: DataFrame = read_csv('minMaxedTrainData.csv')
valSet: DataFrame = read_csv('minMaxedValData.csv')
testSet: DataFrame = read_csv('minMaxedTestData.csv')


yTrain, yVal, yTest = trainSet['readmitted'], valSet['readmitted'], testSet['readmitted']
xTrain, xVal, xTest = trainSet.drop(columns=['readmitted']), valSet.drop(
    columns=['readmitted']), testSet.drop(columns=['readmitted'])


labels = unique(yTrain)
labels.sort()

# %% Grid Search
n_estimators = [100, 200, 400, 500]
max_depths = [8, 24, 58, 78, 86]
learning_rate = [0.1, 0.5, 0.9]
best = ('', 0, 0)
last_best = 0
best_model = None


cols = len(max_depths)

figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)

overfittingRows = len(max_depths)
overfittingCols = len(learning_rate)

figure()
figOverfitting, axsOverfitting = subplots(overfittingRows, overfittingCols, figsize=(
    len(n_estimators)*HEIGHT, overfittingRows * HEIGHT), squeeze=False)
figOverfitting.suptitle('Overfitting Test')
eval_metric = accuracy_score


for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for maxFeatureIndex, f in enumerate(learning_rate):
        yvalues = []
        yTrainMetrics = []

        for n in n_estimators:
            gb = GradientBoostingClassifier(
                n_estimators=n, max_depth=d, learning_rate=f, min_samples_split=250, min_samples_leaf=250)
            gb.fit(xTrain, yTrain)
            prdY = gb.predict(xVal)
            prdYTrain = gb.predict(xTrain)
            yvalues.append(eval_metric(yVal, prdY))
            yTrainMetrics.append(eval_metric(yTrain, prdYTrain))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = gb

        axis = set_elements(
            ax=axsOverfitting[k, maxFeatureIndex], title='LearningRate = ' + str(f) + ', MaxDepth = ' + str(d), ylabel=eval_metric, xlabel='NumOfEstimators')
        axis.plot(n_estimators, yTrainMetrics, color='orange',
                  linewidth=2, markersize=12, label='Train')
        axis.plot(n_estimators, yvalues, color='blue',
                  linewidth=2, markersize=12, label='Validation')
        axis.set_ylim((0, 1))
        axis.legend(loc="lower right")

        values[f] = yvalues
    figOverfitting.savefig(f'images/{file_tag}_gb_overfittingStudy.png')

    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boosting with max_depth={d}',
                        xlabel='nr estimators', ylabel='accuracy', percentage=True)

fig.savefig(f'images/{file_tag}_gb_study.png')
show()
print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f with min_samples_split and min_samples_leaf equal to 250, in order to avoid overfitting' %
      (best[0], best[1], best[2], last_best))

# %% Best Model params

prd_trn = best_model.predict(xTrain)
prd_tst = best_model.predict(xTest)
plot_evaluation_results(labels, yTrain, prd_trn, yTest, prd_tst)
savefig(f'images/{file_tag}_gb_best.png')
show()

# %% Get Features Importance

train = xTrain
variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
stdevs = std(
    [tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
elems = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure(figsize=(4, 8))
horizontal_bar_chart(elems, importances[indices], stdevs[indices],
                     title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/{file_tag}_gb_ranking.png')
tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

# %%
