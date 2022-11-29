from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
import sys
sys.path.insert(1, '/ds_charts')
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
filename = 'diabetesDataset/diabetic_data.csv'
data = read_csv(filename, na_values='?')
data = data.drop(['encounter_id'], axis=1)
data = data.drop(['patient_nbr'], axis=1)
data_num = data.drop(['admission_type_id'], axis=1)
data_num = data_num.drop(['discharge_disposition_id'], axis=1)
data_num = data_num.drop(['admission_source_id'], axis=1)

# numeric_vars = get_variable_types(data_num)['Numeric']
# if [] == numeric_vars:
#     raise ValueError('There are no numeric variables.')

# rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# for i in range(len(numeric_vars)):
#     var1 = numeric_vars[i]
#     for j in range(i+1, len(numeric_vars)):
#         var2 = numeric_vars[j]
#         axs[i, j-1].set_title("%s x %s"%(var1,var2))
#         axs[i, j-1].set_xlabel(var1)
#         axs[i, j-1].set_ylabel(var2)
#         axs[i, j-1].scatter(data[var1], data[var2])
# savefig(f'diabetesDataset/lab1/images/sparsity/sparsity_study_numeric.png')
# show()


######### mal tenha as categoricas #######
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

data = data.drop(['weight'], axis=1)
data = data.drop(['medical_specialty'], axis=1)
data = data.drop(['payer_code'], axis=1)
data = data.drop(['diag_1'], axis=1)
data = data.drop(['diag_2'], axis=1)
data = data.drop(['diag_3'], axis=1)
data = data.drop(['race'], axis=1)

symbolic_vars = get_variable_types(data)['Numeric']
symbolic_vars = symbolic_vars + get_variable_types(data)['Symbolic']

rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    print(var1)
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        print(var2)
        #axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'diabetesDataset/lab1/images/sparsity/sparsity_study_symbolic_all_teste_com_nomes.png')
show()