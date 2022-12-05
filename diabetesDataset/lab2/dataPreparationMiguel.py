# %%# - Imports

import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots
from ds_charts import bar_chart, get_variable_types, plot_evaluation_results
import sklearn
import imblearn
import numpy as np

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
data = pd.read_csv('../lab1/allEncoded_Diabetes_Data.csv')
data.drop(['encounter_id', 'patient_nbr', 'Unnamed: 0'], axis=1, inplace=True) ## Dropping ids



# %% Scalling Z-Score
def zScoreScalling(data: pd.DataFrame) -> pd.DataFrame:
    
    data = data.copy()
    variableTypes = get_variable_types(data)
    numericVars = variableTypes['Numeric']
    booleanVars = variableTypes['Binary']
    symbolicVars = variableTypes['Symbolic']

    numericData = data[numericVars]
    booleanData = data[booleanVars]
    symbolicData = data[symbolicVars]

    scaler = sklearn.preprocessing.StandardScaler().fit(numericData) #Only numeric can be scalled
    scalledNumericData = pd.DataFrame(scaler.transform(numericData), index=data.index, columns=numericVars)
    zScoreData = pd.concat([scalledNumericData, symbolicData, booleanData], axis=1)

    return zScoreData, scaler

# %% Scalling MinMax
def minMaxScalling(data : pd.DataFrame) -> pd.DataFrame:

    
    data = data.copy()
    variableTypes = get_variable_types(data)
    numericVars = variableTypes['Numeric']
    booleanVars = variableTypes['Binary']
    symbolicVars = variableTypes['Symbolic']

    numericData = data[numericVars]
    booleanData = data[booleanVars]
    symbolicData = data[symbolicVars]

    scaler = sklearn.preprocessing.MinMaxScaler().fit(numericData)
    scalledNumericData = pd.DataFrame(scaler.transform(numericData), index=data.index, columns=numericVars)
    minMaxData = pd.concat([scalledNumericData, symbolicData, booleanData], axis=1)
    
    return minMaxData, scaler

# %% Plots

ig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
zScoreScalling(data).boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
minMaxScalling(minMaxScalling).boxplot(ax=axs[0, 2])
show()

# %% Dataset balancing
data = pd.read_csv('../lab1/allEncoded_Diabetes_Data.csv')

# %% Dataset Balancing (SMOTE)
def smoteBalancing(trainingData: pd.DataFrame, classLabel: str) -> pd.DataFrame:
    trainingData = trainingData.copy()
    RANDOM_STATE = 42
    smote = imblearn.over_sampling.SMOTE(random_state=RANDOM_STATE)
    yData = trainingData[classLabel]
    xData = trainingData.drop(columns=[classLabel])
    xData, yData = smote.fit_resample(xData, yData)
    trainingData = pd.concat([xData, yData], axis=1)

    return trainingData

# %% Dataset Balacing (Smoothen Class Weights)

def smoothClassWeights(data : pd.DataFrame, classLabel: str) -> dict :

    trainingData = trainingData.copy()
    labelsFreqs = data[classLabel].value_counts().to_dict()
    total = np.sum(labelsFreqs.values())
    labels = labelsFreqs.keys()
    weights = dict()

    for label in labels:
        weights[label] = np.log((0.15*total)/float(labelsFreqs[label]))
        weights[label] = weights[label] if weights[label] > 1 else 1
    
    return weights



# %% Dataset Balancing (Tomek's Link + SMOTE)

def smoteTomeksBalancing(trainingData: pd.DataFrame, classLabel: str) -> pd.DataFrame:

    trainingData = trainingData.copy()
    RANDOM_STATE = 42

    tomeks = imblearn.under_sampling.TomekLinks()
    smote = imblearn.over_sampling.SMOTE(random_state=RANDOM_STATE)

    yData = trainingData[classLabel]
    xData = trainingData.drop(columns=[classLabel])

    xData, yData = tomeks.fit_resample(xData, yData)
    xData, yData = smote.fit_resample(xData, yData)

    trainingData = pd.concat([xData, yData], axis=1)

    return trainingData

# %% testing the scalling alternatives

def scallingEvaluator(data: pd.DataFrame, classLabel: str):

    data = data.copy()
    classifiersList = []
    variableTypes = get_variable_types(data)
    numericVars = variableTypes['Numeric']
    booleanVars = variableTypes['Binary']
    symbolicVars = variableTypes['Symbolic']

    RANDOM_STATE = 42   
    # Getting x and y
    yData = data[classLabel]
    labels = pd.unique(yData)
    xData = data.drop(columns=[classLabel])
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(xData, yData, train_size=0.7, random_state=RANDOM_STATE)

    # zScore
    xTrainZScore, scaler = zScoreScalling(xTrain)
    xTestZScore = pd.DataFrame(scaler.transform(
        xTestZScore[numericVars]), index=xTest.index, columns=numericVars)
    xTestZScore = pd.concat(
        [xTestZScore, xTest[symbolicVars], xTest[booleanVars]], axis=1)
        
    # Fitting the model to the training data
    naiveBayesClassifier = sklearn.naive_bayes.GaussianNB()
    naiveBayesClassifier.fit(xTrainZScore, yTrain)
    classifiersList.append((naiveBayesClassifier, 'zScoreNB', xTrainZScore, xTestZScore))

    knnClassifier = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=50, metric='manhattan')
    knnClassifier.fit(xTrainZScore, yTrain)
    classifiersList.append((knnClassifier, 'zScoreKNN', xTrainZScore, xTestZScore))

    # Min max

    xTrainMinMax, scaler = minMaxScalling(xTrain)
    xTestMinMax = pd.DataFrame(scaler.transform(
        xTestMinMax[numericVars]), index=xTest.index, columns=numericVars)
    xTestMinMax = pd.concat(
        [xTestMinMax, xTest[symbolicVars], xTest[booleanVars]], axis=1)

    # Fitting the model to the training data
    naiveBayesClassifier = sklearn.naive_bayes.GaussianNB()
    naiveBayesClassifier.fit(xTrainMinMax, yTrain)
    classifiersList.append((naiveBayesClassifier, 'minMaxNB', xTrainMinMax, xTestMinMax))

    knnClassifier = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=50, metric='manhattan')
    knnClassifier.fit(xTrainMinMax, yTrain)
    classifiersList.append(
        (knnClassifier, 'minMaxKNN', xTrainMinMax, xTestMinMax))

    for classifier, name, xTrain, yTrain in zip(classifiersList):

        predictedTrainY = classifier.predict(xTrain) # overfitted
        predictedTestY = classifier.predict(xTest) # hopefully not overfitted
        plot_evaluation_results(labels, yTrain, predictedTrainY, yTest, predictedTestY)
        savefig('images/scalling/{name}Eval.png')
        show()
    

def balancingEvaluator(data: pd.DataFrame, classLabel: str):

    RANDOM_STATE = 42
    # Getting x and y
    yData = data[classLabel]
    labels = pd.unique(yData)
    xData = data.drop(columns=[classLabel])
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(
        xData, yData, train_size=0.7, random_state=RANDOM_STATE)
    
    for name in ['SMOTE', 'TomeksAndSMOTE', 'SmoothenClassWeights']:

        if name == 'SMOTE':

            xTrainSMOTE = smoteBalancing(xTrain, classLabel) 
            classifier = sklearn.naive_bayes.GaussianNB()
            classifier.fit(xTrainSMOTE, yTrain)

            predictedTrainY = classifier.predict(xTrainSMOTE)  # overfitted
            predictedTestY = classifier.predict(xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}NBEval.png')
            show()

            classifier = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=50, metric='manhattan')
            classifier.fit(xTrainSMOTE, yTrain)
            predictedTrainY = classifier.predict(xTrainSMOTE)  # overfitted
            predictedTestY = classifier.predict(
                xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}KNNEval.png')
            show()

        elif name == 'TomeksAndSMOTE':

            xTrainTomek = smoteTomeksBalancing(xTrain, classLabel)
            classifier = sklearn.naive_bayes.GaussianNB()
            classifier.fit(xTrainTomek, yTrain)

            predictedTrainY = classifier.predict(xTrainTomek)  # overfitted
            predictedTestY = classifier.predict(xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}NBEval.png')
            show()

            classifier = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=50, metric='manhattan')
            classifier.fit(xTrainTomek, yTrain)
            predictedTrainY = classifier.predict(xTrainTomek)  # overfitted
            predictedTestY = classifier.predict(
                xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}KNNEval.png')
            show()
        elif name == 'SmoothenClassWeights':
            weights = smoothClassWeights(xTrain, classLabel)

            xTrain = smoteTomeksBalancing(xTrain, classLabel)
            classifier = sklearn.naive_bayes.GaussianNB(class_weights = weights)
            classifier.fit(xTrain, yTrain)

            predictedTrainY = classifier.predict(xTrain)  # overfitted
            predictedTestY = classifier.predict(xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}NBEval.png')
            show()

            classifier = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=50, metric='manhattan', class_weights = weights)
            classifier.fit(xTrain, yTrain)
            predictedTrainY = classifier.predict(xTrain)  # overfitted
            predictedTestY = classifier.predict(
                xTest)  # hopefully not overfitted
            plot_evaluation_results(
                labels, yTrain, predictedTrainY, yTest, predictedTestY)
            savefig('images/scalling/{name}KNNEval.png')
            show()
