#%% Imports
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

#%% Load Dataset

data = pd.read_csv('mv_glocose.csv')

#%% Scalling ZScore
def zScoreScalling(data: pd.DataFrame) -> pd.DataFrame:
    
    data = data.copy()

    # Remove alredy well behaved dimensions
    date = data['Date']

    data.drop(columns=['Date'], inplace=True)

        
    scaler = preprocessing.StandardScaler().fit(data)
    scalledData = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    zScoreData = pd.concat([date, scalledData], axis=1)

    return zScoreData, scaler



# %% get scalled Data
zScoreData, _ = zScoreScalling(data)

zScoreData.to_csv('preTransformationsGlucose.csv', index = False)
# %%
