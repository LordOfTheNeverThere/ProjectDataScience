# %%# - Imports

from seaborn import heatmap
from matplotlib.pyplot import figure, savefig, show, title
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, tight_layout
from ds_charts import bar_chart, dummify

register_matplotlib_converters()

# kaggleOGData = pd.read_csv('../kaggleDataset.csv')
data = pd.read_csv('../diabetic_data.csv', na_values='?')

data.shape
# %% #


figure(figsize=(4, 2))
values = {'num Of Records': data.shape[0], 'num Of Variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()),
          title='Comparison between number of vars and dims')
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savefig('images/records_variables.png')

show()

# %%
data.dtypes
# %%


def get_variable_types(df) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int64' or df[c].dtype == 'float64':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)
    return variable_types


variable_types = get_variable_types(data)

counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4, 2))
bar_chart(list(counts.keys()), list(counts.values()),
          title='Frequency in var type')
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savefig('images/variable_types.png')
show()

# %%
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Number of Missing values per var',
          ylabel='Number of Missing values', rotation=True)
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savefig('images/mv.png')
show()

# %% Correlation Analysis
corr_mtx = abs(data.corr())
print(corr_mtx)

# %%

fig = figure(figsize=[12, 12])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns,
        yticklabels=corr_mtx.columns, annot=True, cmap='Reds')
title('Correlation analysis')
savefig(f'images/correlation_analysis.png')
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
show()

# %% Get ICD9 Categorised Dataset
# Dummification/Encoding

data = pd.read_csv('../lab2/diabetic_data_ICD9Cats.csv')


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
    data.drop(columns=['weight', 'medical_specialty',
                       'payer_code', 'Unnamed: 0'], inplace=True)
    # Drop out all records with missing values
    data.dropna(inplace=True)
    varsToDummify = ['race']
    df = dummify(data, varsToDummify)

    data = df
    data['gender'] = data['gender'].apply(genderEncoder)
    data['age'] = data['age'].apply(ageEncoder)
    data['max_glu_serum'] = data['max_glu_serum'].apply(max_glu_serumEncoder)
    data['A1Cresult'] = data['A1Cresult'].apply(a1c_resultEncoder)
    data[data.columns[35:58]] = data[data.columns[35:58]].applymap(drugEncoder)
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


data = allEncoder(data)
data.to_csv('allEncoded_Diabetes_Data.csv')
# %%
# Full correlation Analysis

# First We drop all vars which hold no variation, which are constant
nunique = data.nunique()
colsToDrop = nunique[nunique == 1].index
data.drop(colsToDrop, axis=1, inplace=True)
variable_types = get_variable_types(data) #Get the right types in place

corr_mtx = abs(data[variable_types['Numeric']].corr())
print(corr_mtx)

fig = figure(figsize=[40, 40])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns,
        yticklabels=corr_mtx.columns, annot=True, cmap='Greys')
title('Correlation analysis')
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savefig(f'images/allDims_correlation_analysis.png')
show()
# %%
#Variable Frequency all dims
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4, 2))
bar_chart(list(counts.keys()), list(counts.values()),
          title='Frequency in var type in final encoded dataset')
tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savefig('images/allDims_variable_types.png')
show()

