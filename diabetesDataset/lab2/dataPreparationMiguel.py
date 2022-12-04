# %%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots
from ds_charts import bar_chart, get_variable_types
import sklearn

register_matplotlib_converters()

# %% Data Preparation (Creating ICD-9 Categories)
data = pd.read_csv('../diabetic_data.csv', na_values='?')

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
    print(icd9Array)
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

diagDf = data[['diag_1', 'diag_2', 'diag_3']]
data.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)
newFeatures = diseaseEncoder(diagDf)

for index, newFeature in enumerate(newFeatures):
    data.insert(17 + index, newFeature ,newFeatures[newFeature]) ## Data with the new categories
# %%
data.to_csv('diabetic_data_ICD9Cats.csv')

# %% Scalling

variableTypes = get_variable_types(data)
numericVars = variableTypes['Numeric']
booleanVars = variableTypes['Binary']
symbolicVars = variableTypes['Symbolic']

numericData = data[numericVars]
booleanData = data[booleanVars]
symbolicData = data[symbolicVars]

# %% Scalling Z-Score

scaler = sklearn.preprocessing.StandardScaler().fit(numericData) #Only numeric can be scalled
scalledNumericData = pd.DataFrame(scaler.transformation(numericData), index=data.index, columns=numericVars)
zScoreData = pd.concat([scalledNumericData, symbolicData, booleanData], axis=1)
#zScoreData.to_csv('putAGoodNamezScore.csv')

# %% Scalling MinMax

scaler = sklearn.preprocessing.MinMaxScaler().fit(numericData)
scalledNumericData = pd.DataFrame(scaler.transform(numericData), index=data.index, columns=numericVars)
minMaxData = pd.concat([scalledNumericData, symbolicData, booleanData], axis=1)
#minMaxData.to_csv('putAGoodNameMinMac.csv', index = False)

# %% Plots

ig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
zScoreData.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
minMaxData.boxplot(ax=axs[0, 2])
show()
