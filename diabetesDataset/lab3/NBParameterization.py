#%% initialize
import numpy as np
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart
from sklearn.model_selection import train_test_split

data = read_csv('minMaxedData.csv')


#%%
target = 'readmitted'

y = data.pop(target).values
X = data.values
labels: np.ndarray = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify = y)


#%%

from ds_charts import plot_evaluation_results
from sklearn.naive_bayes import GaussianNB, MultinomialNB

clf = BernoulliNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)

plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)

# savefig('images/nb_bestresult.png')
#%%
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB(),
              # 'CategoricalNB': CategoricalNB() ##### é suposto conseguir usar este?
              }

xvalues = []
yvalues = []
yvalues_2 = []
yvalues_3 = []
yvalues_4 = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))
    yvalues_2.append(recall_score(tstY, prdY, average='weighted'))
    yvalues_3.append(f1_score(tstY, prdY, average='weighted'))
    yvalues_4.append(precision_score(tstY, prdY, average='weighted'))



fig = figure(figsize=(10, 7))
fig.add_subplot(2, 2, 1)
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
fig.add_subplot(2, 2, 2)
bar_chart(xvalues, yvalues_2, title='Comparison of Naive Bayes Models', ylabel='recall', percentage=True)
fig.add_subplot(2, 2, 3)
bar_chart(xvalues, yvalues_3, title='Comparison of Naive Bayes Models', ylabel='f1_score', percentage=True)
fig.add_subplot(2, 2, 4)
bar_chart(xvalues, yvalues_4, title='Comparison of Naive Bayes Models', ylabel='precision', percentage=True)

# savefig('images/nb_study.png')

# %%
