# %% iniciation

from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show, tight_layout
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

target = 'readmitted'
data: DataFrame = read_csv('minMaxedData.csv')

y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.8, stratify=y)

# %%  test 2

target = 'readmitted'
train: DataFrame = read_csv('minMaxedTrainData.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('minMaxedValData.csv')
valY: ndarray = test.pop(target).values
valX: ndarray = test.values


# %% mlp results
HEIGHT: int = 4
lr_type = ['constant', 'invscaling', 'adaptive']
max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
learning_rate = [.1, .5, .9]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(lr_type)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    print(k/len(lr_type))
    d = lr_type[k]
    values = {}
    for lr in learning_rate:
        print(lr)
        yvalues = []
        for n in max_iter:
            print(n)
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                learning_rate_init=lr, max_iter=n, verbose=False)
            mlp.fit(trnX, trnY)
            prdY = mlp.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = mlp
        values[lr] = yvalues
    multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                           xlabel='mx iter', ylabel='accuracy', percentage=True)
savefig('images/mlp_study_tts.png')
print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')
# %% plot best results

## Best results with lr_type=adaptive, 
# learning rate=0.5 and 2500 max iter, 
# with accuracy=0.5834725361108382

# test: DataFrame = read_csv('minMaxedTestData.csv')
# tstY: ndarray = test.pop(target).values
# tstX: ndarray = test.values
from matplotlib.pyplot import figure, subplots, savefig, show, tight_layout
best_model = MLPClassifier(activation='logistic', solver='sgd', learning_rate='adaptive',
                                learning_rate_init=.5 , max_iter=2500, verbose=False)
best_model.fit(trnX, trnY)
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
#  plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig('images/mlp_best.png')


# loss curves
import matplotlib.pyplot as plt

# plt.plot(best_model.loss_curve_)
# plt.plot(best_model.validation_scores_)


#%% overfitting study

from ds_charts import plot_overfitting_study

lr_type = 'adaptive' # try other lr types
lr = 0.9 # try other lr values
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in max_iter:
    mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
    mlp.fit(trnX, trnY)
    prd_tst_Y = mlp.predict(tstX)
    prd_trn_Y = mlp.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr episodes', ylabel=str(eval_metric))
savefig('images/mlp_overfitting_adpt_9.png')
