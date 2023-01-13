from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score

file_tag = 'drought'
filename = 'drought'
target = 'class'

train: DataFrame = read_csv('../Data/Balancing/drought_scaled_smote.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'../Data/TrainTest/drought_prepared_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
# n_estimators = [5]
max_depths = [5, 10, 25, 50]
# max_depths = [5]

max_features = [.3, .5, .7, 1]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(max_depths)):

    d = max_depths[k]
    values = {}
    for f in max_features:
        print(f)
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = rf

        values[f] = yvalues
    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                           xlabel='nr estimators', ylabel='accuracy', percentage=True)
savefig(f'images/RandomForests/{file_tag}_rf_study2.png')
# show()
print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))


prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/RandomForests/{file_tag}_rf_best2.png')
# show()
#
#
# from numpy import std, argsort
#
# variables = train.columns
# importances = best_model.feature_importances_
# stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
# indices = argsort(importances)[::-1]
# elems = []
# for f in range(len(variables)):
#     elems += [variables[indices[f]]]
#     print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')
#
# figure()
# horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
# savefig(f'images/RandomForests/{file_tag}_rf_ranking.pdf')


from ds_charts import plot_overfitting_study

f = 0.7
max_depth = 50
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in n_estimators:
    print(n)
    rf = RandomForestClassifier(n_estimators=n, max_depth=25, max_features=f)
    rf.fit(trnX, trnY)
    prd_tst_Y = rf.predict(tstX)
    prd_trn_Y = rf.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))

