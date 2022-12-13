#%% initialize
import numpy as np
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart
from sklearn.model_selection import train_test_split

data = read_csv('minMaxedData.csv')

data = data.drop(columns=['Unnamed: 0.1','Unnamed: 0'])
target = 'readmitted'

y = data.pop(target).values
X = data.values
labels: np.ndarray = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify = y)

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
    yvalues_2.append(recall_score(tstY, prdY, average='macro'))
    yvalues_3.append(f1_score(tstY, prdY, average='macro'))
    yvalues_4.append(precision_score(tstY, prdY, average='macro'))

figure()
# bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
# bar_chart(xvalues, yvalues_2, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
# bar_chart(xvalues, yvalues_3, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
bar_chart(xvalues, yvalues_4, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)

savefig('images/nb_study_precision.png')
# show()
# %%
