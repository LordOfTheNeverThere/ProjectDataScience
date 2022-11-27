# %%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

data = pd.read_csv('../diabetic_data.csv', na_values='?')
# %% Data Preparation (Creating ICD-9 Categories)


def icd9Sorter(diseaseCodes):

    icd9Array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for diseaseCode in diseaseCodes:
        if diseaseCode < 140:
            icd9Array[0] += 1

        elif diseaseCode < 240:
            icd9Array[1] += 1

        elif diseaseCode < 280:
            icd9Array[2] += 1

        elif diseaseCode < 290:
            icd9Array[3] += 1

        elif diseaseCode < 320:
            icd9Array[4] += 1   

        elif diseaseCode < 390:
            icd9Array[5] += 1    

        elif diseaseCode < 460:
            icd9Array[6] += 1 

        elif diseaseCode < 520:
            icd9Array[7] += 1 

        elif diseaseCode < 580:
            icd9Array[8] += 1 

        elif diseaseCode < 630:
            icd9Array[9] += 1 

        elif diseaseCode < 680:
            icd9Array[10] += 1 

        elif diseaseCode < 710:
            icd9Array[11] += 1 

        elif diseaseCode < 740:
            icd9Array[12] += 1 

        elif diseaseCode < 760:
            icd9Array[13] += 1 

        elif diseaseCode < 780:
            icd9Array[14] += 1

        elif diseaseCode < 800:
            icd9Array[15] += 1

        elif diseaseCode < 1000:
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
        'ObstetricComplications': [],
        'dermatoPatologies': [],
        'locomotivePatologies': [],
        'congenitalAnomalies': [],
        'PrenatalPatologies': [],
        'unknownPatologies': [],
        'injuryAndPoisoning': []
    }
    for sample in diagnosticsDf:
        icd9Frequencies = icd9Sorter(sample)
        for icd9Frequency, key in zip(icd9Frequencies, icd9Dict):
            icd9Dict[key].append(icd9Frequency)
