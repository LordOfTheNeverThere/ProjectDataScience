import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, savefig, show
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from ds_charts import plot_line

#### READ DATA

index_col = 'date'
target = 'QV2M'
file_tag="dtimeseries"
data = read_csv('../Data/TimeSeries/drought.forecasting_dataset.csv', index_col="date", sep=',', decimal='.', parse_dates=True, dayfirst=True, usecols = ["date", target])

#### FAZER OS TRAINING SETS

train, test = split_dataframe(data, trn_pct=0.70)

measure = 'R2'
flag_pct = False
eval_results = {}

#### CLASSIFIER
class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

measure = 'R2'
flag_pct = False
eval_results = {}

R2results = [0.9520385025732975, 0.8861878042854796, 0.837520954090581, 0.8061512144152332, 0.7841120683275635, 0.7667093658536439, 0.7515339955865363, 0.7392194937039369, 0.7296405824124139, 0.7228925062079575, 0.7184167677152306, 0.7152666934217501, 0.7124198586293816, 0.7093823668341785, 0.7064296264375902, 0.7033132569972191, 0.6995755648351587, 0.6953378422889218, 0.6910637814943968, 0.6865324485523041, 0.6817097366531114, 0.6774456461477114, 0.6739921708423054, 0.6706557981955108, 0.6673154191538375, 0.6640844216448324, 0.6610032697166279, 0.6580418386283338, 0.65494187659653, 0.6513496818468465, 0.6471532552142172, 0.6425536708939019, 0.6380125541910819, 0.6338457007695968, 0.6298835642641505, 0.6258626870394611, 0.6215315023894065, 0.6169467752132697, 0.6122865665088681, 0.6076836527323962, 0.6028019791438775, 0.5973557031905152, 0.5916141476499885, 0.5861926384525916, 0.5812521835423594, 0.5765711322510411, 0.5716681736456948, 0.5666601469247075, 0.5616261555378772, 0.5564867696730211, 0.551330861694308, 0.5462057175187187, 0.540854701890334, 0.5351900008731276, 0.5294706781692893, 0.5238138058853412, 0.5181418727134042, 0.5123608428742945, 0.5064356282249607, 0.5005486822249989, 0.4947292974174309, 0.48888788996058674, 0.4830253328450266, 0.47720270004071974, 0.4712874141713965, 0.4652046006919992, 0.4591483344912115, 0.45320977359052994, 0.4472171958926999, 0.44116489533905334, 0.43496288779886727, 0.4287517226600599, 0.42253509927104993, 0.4162807425898911, 0.4101787126187938, 0.40432833560341175, 0.3986315324003118, 0.39276295341108614, 0.38657432799582103, 0.38023948780725736, 0.37376611482157995, 0.36712162874702836, 0.3604026397275125, 0.35382671867552573, 0.34746826758318694, 0.34127377579520246, 0.3350774211606141, 0.3287406701337038, 0.32239563629412193, 0.3161157206039258, 0.309750321933284, 0.30320761924978157, 0.2965102481224955, 0.2897852439459212, 0.28316251498078093, 0.2766087767326483, 0.2700679602924668, 0.26347111189396866]
best = (0,0)

# for w in range (2, 5):
#     print(w)
#     fr_mod = RollingMeanRegressor(w)
#     fr_mod.fit(train)
#     prd_trn = fr_mod.predict(train)
#     prd_tst = fr_mod.predict(test)
#
#     eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
#     plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, w)
#     savefig(f'images/RollingMean/{file_tag}_rollingMean_eval_{w}.png')
#
#     plot_forecasting_series(train, test, prd_trn, prd_tst, f'Rolling Mean Plots Window = {w}', x_label=index_col, y_label=target)
#     savefig(f'images/RollingMean/{file_tag}_rollingMean_plots_{w}.png')


# print(R2results)
plot_line([x for x in range(2,100)], R2results, title = "R2 Score for different Window Sizes", xlabel = "Window Size", ylabel= "R2")
show()
savefig(f'images/RollingMean/{file_tag}_rollingMean_study.png')




