from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

# READING

filename = '../Data/drought_scaled.csv'
file = "drought_scaled"
original = read_csv(filename, sep=',', decimal='.')

# HISTOGRAM

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

rows, cols = choose_grid(1)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(1):
    axs[i, j].set_title('Histogram')
    axs[i, j].set_xlabel("Class")
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(original["class"].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig(f'images/Balancing/{file}_histogram.png')
show()

# DATA BALANCE

class_var = 'class'
target_count = original[class_var].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
#ind_positive_class = target_count.index.get_loc(positive_class)
print('No Drought=', positive_class, ':', target_count[positive_class])
print('Drought=', negative_class, ':', target_count[negative_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
values = {'Original': [target_count[positive_class], target_count[negative_class]]}

# POSITIVE CLASS = NO DROUGHT

figure()
bar_chart(["Drought", "No Drought"], target_count.values, title='Class balance')
savefig(f'images/Balancing_{file}_data_balance.png')
show()

# SPLIT

df_positives = original[original[class_var] == positive_class]
df_negatives = original[original[class_var] == negative_class]

# UNDERSAMPLING

print("\nUNDERSAMPLING \n")
from pandas import concat, DataFrame

df_neg_sample = DataFrame(df_negatives.sample(len(df_positives))) #
df_under = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f'../Data/Balancing/{file}_under.csv', index=False) # changed
values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
print('No Drought=', positive_class, ':', len(df_positives))
print('Drought=', negative_class, ':', len(df_neg_sample))
print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')

#OVERSAMPLING

print("\nOVERSAMPLING \n")
from pandas import concat, DataFrame

df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_over = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f'../Data/Balancing/{file}_over.csv', index=False)
values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
print('No Drought=', positive_class, ':', len(df_pos_sample))
print('Drought=', negative_class, ':', len(df_negatives))
print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')

# SMOTE
print("\nSMOTE \n")

from pandas import Series
from imblearn.over_sampling import SMOTE
RANDOM_STATE = 42

smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
y = original.pop(class_var).values
X = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [class_var]
df_smote.to_csv(f'../Data/Balancing/{file}_smote.csv', index=False)

smote_target_count = Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
print('No Drought=', positive_class, ':', smote_target_count[positive_class])
print('Drought=', negative_class, ':', smote_target_count[negative_class])
print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

from matplotlib.pyplot import figure, show, savefig
from ds_charts import multiple_bar_chart

figure()
multiple_bar_chart(["Drought", "No Drought"], values, title='Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/Balancing/{file}_smote.png')
show()
