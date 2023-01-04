# %%

from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT

index_multi = 'Date'
target_multi = 'Glucose'
data_multi = read_csv('dailyGlucose.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='consumption', title=target_multi)
plot_series(data_multi['Insulin'])
xticks(rotation = 45)
show()


# %% missing values
import pandas as pd
data = read_csv('../glucose.csv', na_values='')

data['Date'] = pd.to_datetime(data['Date'], format = "%d/%m/%Y %H:%M")
data = data.set_index('Date')


mv = {}
for var in data_multi:
    nr = data_multi[var].isna().sum()
    if nr > 0:
        mv[var] = nr
print(mv)

data.head()

def missing_values_handle(dataframe):
    hours = dataframe.index.hour.unique()
    df1 = dataframe.copy()
    medians = []

    for x in hours:
        medians.append(df1[df1.index.hour == x]['Insulin'].mean())
    
    df1['Insulin'] = df1['Insulin'].fillna(-1)

    for x in range(len(hours)):
        if sum(df1['Insulin'] == -1) != 0:
            df1.loc[(df1.index.hour == hours[x]) & (df1['Insulin'] == -1), 'Insulin'] = medians[x]
            

    return df1

data = missing_values_handle(data)

data.to_csv('mv_glocose.csv')



# %% smoothing
from pandas import read_csv, Series, to_datetime
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT

data = read_csv('preTransformationsGlucose.csv', na_values='')
data['Date'] = to_datetime(data['Date'], format = "%Y/%m/%d %H:%M")
data = data.set_index('Date')

index_multi = 'Date'
target_multi = 'Glucose'

WIN_SIZE = 80
rolling_multi = data.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(smooth_df_multi[target_multi], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='values')
plot_series(smooth_df_multi['Insulin'], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='values')
xticks(rotation = 45)
savefig('images/transformation/smoothing_80.png')


WIN_SIZE = 12
rolling_multi = data.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(smooth_df_multi[target_multi], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='values')
plot_series(smooth_df_multi['Insulin'], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='values')
xticks(rotation = 45)
savefig('images/transformation/smoothing_12.png')

# %% aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df



figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data, index_multi, 'D')
plot_series(agg_multi_df[target_multi], title='Glucose - Daily values', x_label='timestamp', y_label='values')
plot_series(agg_multi_df['Insulin'], title='Glucose - Daily values', x_label='timestamp', y_label='values')
xticks(rotation = 45)
savefig('images/transformation/aggregation_d.png')

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_multi_df = aggregate_by(data, index_multi, 'W')
# plot_series(agg_multi_df[target_multi], title='Glucose - Weekly values', x_label='timestamp', y_label='values')
# plot_series(agg_multi_df['Insulin'], title='Glucose - Weekly values', x_label='timestamp', y_label='values')
# xticks(rotation = 45)
# savefig('images/transformation/aggregation_w.png')

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_multi_df = aggregate_by(data, index_multi, 'M')
# plot_series(agg_multi_df[target_multi], title='Glucose - Monthly values', x_label='timestamp', y_label='values')
# plot_series(agg_multi_df['Insulin'], title='Glucose - Monthly values', x_label='timestamp', y_label='values')
# xticks(rotation = 45)
# savefig('images/transformation/aggregation_m.png')

# %%
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from pandas import DataFrame
measure = 'R2'
flag_pct = False
eval_results = {}
def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test
########### só está para univariados ######### é preciso mudar
train, test = split_dataframe(agg_multi_df, trn_pct=0.75)

class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)


# %%
eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

# %%

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'images/transformation/persistence_eval.png')
savefig('images/transformation/persistence_eval_mon.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, 'images/transformation/_persistence_plots.png', x_label='Date', y_label='Glucose')
savefig('images/transformation/_persistence_plots_mon.png')


# %%
