# %%
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()
filename = '../diabetic_data.csv'
data = read_csv('../diabetic_data.csv', na_values='?')

# %%
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, dummify

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
df_2.reset_index(drop=True, inplace=True)
print(df_2.shape)
print(df_2.columns.get_loc('metformin-pioglitazone'))

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
print(numeric_vars)
print(symbolic_vars)
print(binary_vars)
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

# %%

# encoding
import pandas as pd
def genderEncoder(gender): return 1 if (gender == 'Male') else 0


def ageEncoder(age): return 10 if (age[-3:-1] == '10') else 20 if (age[-3:-1] == '20') else 30 if (age[-3:-1] == '30') else 40 if (age[-3:-1] == '40') else 50 if (
    age[-3:-1] == '50') else 60 if (age[-3:-1] == '60') else 70 if (age[-3:-1] == '70') else 80 if (age[-3:-1] == '80') else 90 if (age[-3:-1] == '90') else 100


def max_glu_serumEncoder(glucose): return 0 if (glucose == 'None') else 1 if (
    glucose == 'Norm') else 2 if (glucose == '>200') else 3


def a1c_resultEncoder(result): return 0 if (result == 'None') else 1 if (
    result == 'Norm') else 2 if (result == '>7') else 3


def drugEncoder(drug): return 0 if (drug == 'No') else 1 if (
    drug == 'Down') else 2 if (drug == 'Steady') else 3


def changeEncoder(change): return 0 if (change == 'No') else 1
def diabetesMedEncoder(change): return 0 if (change == 'No') else 1


def readmitedEncoder(label): return 0 if (
    label == '<30') else 1 if (label == '>30') else 2


def admissionTypeEncoder(admissionType): return 1 if (
    admissionType == 3) else 2 if (admissionType == 4) else 3 if (admissionType == 2) else 4 if (admissionType == 7) else 5 if (admissionType == 1) else 0  # Null values


def dischargeTypeEncoder(disType): return 1 if (disType == 1) else 2 if (disType == 6) else 3 if (disType == 8) else 4 if (disType == 15 or disType == 27 or disType == 4) else 5 if (disType == 3) else 6 if (disType == 24) else 7 if (disType == 30 or disType == 7) else 8 if (disType == 12) else 9 if (disType == 17) else 10 if (
    disType == 16) else 11 if (disType == 5) else 12 if (disType == 23) else 13 if (disType == 29) else 14 if (disType == 2) else 15 if (disType == 9) else 16 if (disType == 13) else 17 if (disType == 14) else 18 if (disType == 19) else 19 if (disType == 20) else 20 if (disType == 21) else 21 if (disType == 22) else 22 if (disType == 28) else 0


def admissionSourceEncoder(source): return 1 if (source == 2) else 2 if (source == 1) else 3 if (source == 11) else 4 if (source == 10) else 5 if (source == 6) else 6 if (source == 5) else 7 if (source == 19) else 8 if (source == 18) else 9 if (source == 25) else 10 if (
    source == 23) else 11 if (source == 14) else 12 if (source == 24) else 13 if (source == 4) else 14 if (source == 22) else 15 if (source == 12) else 16 if (source == 13) else 17 if (source == 26) else 18 if (source == 8) else 19 if (source == 7) else 0


def allEncoder(df):
    data = df.copy()
    # drop sparse dims
    # Drop out all records with missing values
    varsToDummify = ['race']
    df = dummify(data, varsToDummify)
    in_first_med = df.columns.get_loc('metformin')
    break_point = df.columns.get_loc('readmitted')
    in_last_med = df.columns.get_loc('metformin-pioglitazone')

    data = df
    data['gender'] = data['gender'].apply(genderEncoder)
    data['age'] = data['age'].apply(ageEncoder)
    data['max_glu_serum'] = data['max_glu_serum'].apply(max_glu_serumEncoder)
    data['A1Cresult'] = data['A1Cresult'].apply(a1c_resultEncoder)
    data[data.columns[in_first_med:break_point]] = data[data.columns[in_first_med:break_point]].applymap(drugEncoder)
    data[data.columns[break_point+1:in_last_med+1]] = data[data.columns[break_point+1:in_last_med+1]].applymap(drugEncoder)
    data['change'] = data['change'].apply(changeEncoder)
    data['diabetesMed'] = data['diabetesMed'].apply(diabetesMedEncoder)
    data['readmitted'] = data['readmitted'].apply(readmitedEncoder)
    data['admission_type_id'] = data['admission_type_id'].apply(
        admissionTypeEncoder)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].apply(
        dischargeTypeEncoder)
    data['admission_source_id'] = data['admission_source_id'].apply(
        admissionSourceEncoder)

    return data


df_3 = allEncoder(df_3)

def icd9Sorter(diseaseCodes):
    
    icd9Array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for diseaseCode in diseaseCodes:
        if type(diseaseCode) == str and (diseaseCode[0] == 'V' or diseaseCode[0] == 'E'):
            icd9Array[-1] += 1
        
        elif float(diseaseCode) < 140:
            icd9Array[0] += 1

        elif float(diseaseCode) < 240:
            icd9Array[1] += 1

        elif float(diseaseCode) < 280:
            icd9Array[2] += 1

        elif float(diseaseCode) < 290:
            icd9Array[3] += 1

        elif float(diseaseCode) < 320:
            icd9Array[4] += 1   

        elif float(diseaseCode) < 390:
            icd9Array[5] += 1    

        elif float(diseaseCode) < 460:
            icd9Array[6] += 1 

        elif float(diseaseCode) < 520:
            icd9Array[7] += 1 

        elif float(diseaseCode) < 580:
            icd9Array[8] += 1 

        elif float(diseaseCode) < 630:
            icd9Array[9] += 1 

        elif float(diseaseCode) < 680:
            icd9Array[10] += 1 

        elif float(diseaseCode) < 710:
            icd9Array[11] += 1 

        elif float(diseaseCode) < 740:
            icd9Array[12] += 1 

        elif float(diseaseCode) < 760:
            icd9Array[13] += 1 

        elif float(diseaseCode) < 780:
            icd9Array[14] += 1

        elif float(diseaseCode) < 800:
            icd9Array[15] += 1

        elif float(diseaseCode) < 1000:
            icd9Array[16] += 1
  
    return icd9Array
        


def diseaseEncoder(diagnosticsDf: pd.DataFrame) -> pd.DataFrame:
    """Will encode the diagnostic vars in a new set of numeric vars, where the ser is composed
    of all icd0 Disease categories

    Args:
        diagnosticsDf (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    diagnosticsDf = diagnosticsDf.copy()
    
    icd9Dict = {
        'infectious\parasiticPatologies': [],
        'neoplasms': [],
        'endocrine/immunity disorders': [],
        'hematoPatologies': [],
        'psychicPatologies': [],
        'neurosensitivePatologies': [],
        'cardioPatologies': [],
        'pneumoPatologies': [],
        'digestivePatologies': [],
        'genitourinaryPatologies': [],
        'obstetricComplications': [],
        'dermatoPatologies': [],
        'locomotivePatologies': [],
        'congenitalAnomalies': [],
        'prenatalPatologies': [],
        'unknownPatologies': [],
        'injuryAndPoisoning': [],
        'externalCauses': []
    }
    for index, sample in diagnosticsDf.iterrows():
        icd9Frequencies = icd9Sorter(
            [sample['diag_1'], sample['diag_2'], sample['diag_3']])
        for icd9Frequency, key in zip(icd9Frequencies, icd9Dict):
            icd9Dict[key].append(icd9Frequency)
    
    newFeaturesDataframe = pd.DataFrame(data=icd9Dict) 

    return newFeaturesDataframe

diagDf = df_3[['diag_1', 'diag_2', 'diag_3']]
df_3.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)
newFeatures = diseaseEncoder(diagDf)

for index, newFeature in enumerate(newFeatures):
    df_3.insert(17 + index, newFeature ,newFeatures[newFeature])

df_3.to_csv('mv_replace_mv.csv')

mv = {}
figure()
for var in df_3:
    nr = df_3[var].isna().sum()
    if nr > 0:
        mv[var] = nr


# %% Bayesian test

import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

data = df_2
target = 'readmitted'

y = data.pop(target).values
X = data.values
labels: np.ndarray = unique(y)
labels.sort()


trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv('mv_drop_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv('mv_drop_test.csv', index=False)



# %%
from sklearn.naive_bayes import GaussianNB
from numpy import ndarray
from sklearn.metrics import confusion_matrix

train: DataFrame = read_csv('mv_drop_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('mv_drop_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)

cnf_mtx_trn = confusion_matrix(tstY, prd_tst)
print(cnf_mtx_trn)

# show()


# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ds_charts
from ds_charts import plot_evaluation_results

plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig('images/mv_drop_nb.png')


# %% KNN classification


from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

clf = knn = KNeighborsClassifier(n_neighbors=11, metric='chebyshev')
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)

ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig('images/mv_drop_knn.png')

# %%
