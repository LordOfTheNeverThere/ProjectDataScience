# %%
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()
data_ = read_csv('mv_replace_mv.csv')
data.shape

# %%
from pandas import DataFrame
from ds_charts import get_variable_types


# drop values
OUTLIER_PARAM: int = 2 # define the number of stdev to use or the IQR scale (usually 1.5)
OPTION = 'iqr'  # or 'stdev'

def determine_outlier_thresholds(summary5: DataFrame, var: str):
    if 'iqr' == OPTION:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:  # OPTION == 'stdev'
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
    median = df[var].median()
    df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

print('data after replacing outliers:', df.describe())
df.to_csv('replace_outliers.csv', index=True)

# %%
import pandas as pd
from ds_charts import get_variable_types
OUTLIER_PARAM: int = 2 # define the number of stdev to use or the IQR scale (usually 1.5)
OPTION = 'iqr'  # or 'stdev'
numeric_vars = get_variable_types(data)['Numeric']
def determine_outlier_thresholds(summary5: pd.DataFrame, var: str):
    if 'iqr' == OPTION:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:  # OPTION == 'stdev'
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

# truncation
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

print('data after truncating outliers:', df.shape)
df.to_csv('truncate_outliers.csv', index=True)


# %%

import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

data = data_
target = 'readmitted'

y = data.pop(target).values
X = data.values
labels: np.ndarray = unique(y)
labels.sort()


trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv('outlier_replace4_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv('outlier_replace4_test.csv', index=False)



train: DataFrame = read_csv('outlier_replace4_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('outlier_replace4_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)

cnf_mtx_trn = confusion_matrix(tstY, prd_tst)




cm_display = ConfusionMatrixDisplay(confusion_matrix = cnf_mtx_trn, display_labels = [0, 1, 2])
cm_display.plot()
plt.title('Accuracy: 0.54')
# savefig('images/outlier_replace3_nb.png')

from sklearn.metrics import accuracy_score
print(accuracy_score(tstY, prd_tst))



eval_metric = accuracy_score
nvalues = [11]
dist = ['chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    y_tst_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        if y_tst_values[-1] > last_best:
            best = (n, d)
            last_best = y_tst_values[-1]
    values[d] = y_tst_values

print(y_tst_values)
cnf_mtx_trn = confusion_matrix(tstY, prd_tst_Y)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cnf_mtx_trn, display_labels = [0, 1, 2])
cm_display.plot()
plt.title('Accuracy: 0.53')
# savefig('images/outlier_replace3_knn.png')


# %%
