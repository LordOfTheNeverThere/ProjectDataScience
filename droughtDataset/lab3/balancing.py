from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart


filename = '../Data/drought_scaled.csv'
file = "drought_scaled"
original = read_csv(filename, sep=',', decimal='.')
class_var = 'class'
target_count = original[class_var].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
#ind_positive_class = target_count.index.get_loc(positive_class)
print('Minority class=', positive_class, ':', target_count[positive_class])
print('Majority class=', negative_class, ':', target_count[negative_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
values = {'Original': [target_count[positive_class], target_count[negative_class]]}

figure()
bar_chart(target_count.index, target_count.values, title='Class balance')
savefig(f'images/Balancing_{file}_balance.png')
show()