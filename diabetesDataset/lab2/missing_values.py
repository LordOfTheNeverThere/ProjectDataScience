# %%
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()
filename = '../diabetic_data.csv'
data = read_csv('../diabetic_data.csv', na_values='?')

# %%
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
#savefig('diabetesDataset/lab2/images/missing_values.png')
print(data.shape)

# %%
 
# 5 types of missing values imputation techniques
# drop variables, drop rows, replace with fixed value, replace mean, replace mean per class

#  1st choice: remove columns with missing values
df = data.copy()
df = data.drop(columns=mv.keys(), inplace=False)
# df.to_csv(f'data/{file}_drop_colu
# mns_mv.csv', index=True)
print('Dropped variables', mv.keys())
print(df.shape)
print(df.columns)
# %%

# 2nd choice: remove weight, payer_code, medical_specialty: +40% valores em falta
# and remove raws with missing values
df_2 = data.copy()
print(df_2.shape)
missings = [c for c in mv.keys() if c in ['weight', 'payer_code', 'medical_specialty']]
df_2 = data.drop(columns=missings, inplace=False)
print('Dropped variables', missings)

df_2 = df_2.dropna()
print(df_2.shape)
print(df_2.columns)

# %%

# 3rd choice: remove weight, payer_code, medical_specialty: +40% valores em falta
# and add mean values to numerical data, most_frequent to symbolic vars, most_frequent to binary
df_3 = data.copy()
missings = [c for c in mv.keys() if c in ['weight', 'payer_code', 'medical_specialty']]
df_3 = data.drop(columns=missings, inplace=False)

from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from ds_charts import get_variable_types
from numpy import nan

tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(df_3)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(df_3[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(df_3[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(df_3[binary_vars]), columns=binary_vars)

df_3 = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df_3.index = data.index
print(df_3.columns)


# %% Bayesian test

from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart

target = 'readmitted'

train = df.copy()
print(train.columns)
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

test = df.copy()
tstY = test.pop(target).values
tstX = test.values

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)


# %%
