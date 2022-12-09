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
positive_class = target_count.idxmax()
negative_class = target_count.idxmin()
#ind_positive_class = target_count.index.get_loc(positive_class)
print('Drought=', positive_class, ':', target_count[positive_class])
print('No Drought=', negative_class, ':', target_count[negative_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
values = {'Original': [target_count[positive_class], target_count[negative_class]]}

figure()
bar_chart(["Drought", "No Drought"], target_count.values, title='Class balance')
savefig(f'images/Balancing_{file}_data_balance.png')
show()

