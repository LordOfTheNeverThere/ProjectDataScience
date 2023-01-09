# %% Imports
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, sliding_window
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import HEIGHT, multiple_line_chart
from sklearn.metrics import r2_score
from torch import zeros
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from pandas import read_csv, DataFrame
from torch import manual_seed, Tensor
from torch.autograd import Variable
from ts_functions import split_dataframe, sliding_window
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#%% Class


class DS_LSTM(Module):
    def __init__(self, input_size, hidden_size, learning_rate, num_layers=1, num_classes=1):
        super(DS_LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                         num_layers=self.num_layers, batch_first=True)
        self.fc = Linear(hidden_size, self.num_classes)
        self.criterion = MSELoss()    # mean-squared error for regression
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        h_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

    def fit(self, trainX, trainY):
        # Train the model
        outputs = self(trainX)
        self.optimizer.zero_grad()
        # obtain the loss function
        loss = self.criterion(outputs, trainY)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, data):
        # Predict the target variable for the input data
        return self(data).detach().numpy()


# %% Data Preprocessing and Loading
target = 'Glucose'
file_tag = 'glucosev0.2'

data = read_csv('../lab6/preTransformationsGlucose.csv')
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d %H:%M")
data = data.set_index('Date')  # Droping Index col

nr_features = len(data.columns)
sc = MinMaxScaler()
data = DataFrame(sc.fit_transform(
    data), index=data.index, columns=data.columns)
manual_seed(50)
train, test = split_dataframe(data, trn_pct=.70)

# %% Data Conversion into tensorn and Training

seq_length = 4
num_epochs = 2000

trnX, trnY = sliding_window(train, seq_length=seq_length)
trnY = trnY[:, 1]
trnX, trnY = Variable(Tensor(trnX)), Variable(Tensor(trnY))
tstX, tstY = sliding_window(test, seq_length=seq_length)
tstY=tstY[:, 1]
tstX, tstY = Variable(Tensor(tstX)), Variable(Tensor(tstY))

my_lstm = DS_LSTM(input_size=2, hidden_size=8, learning_rate=0.001)

for epoch in range(num_epochs+1):
    loss = my_lstm.fit(trnX, trnY)
    if epoch % 500 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss))


#%% Prediction
prd_trn = my_lstm(trnX)
prd_tst = my_lstm(tstX)

print('TRAIN R2=', r2_score(trnY.data.numpy(), prd_trn.data.numpy()))
print('TEST R2=', r2_score(tstY.data.numpy(), prd_tst.data.numpy()))

# %% Study parameterisation


best = ('',  0, 0.0)
last_best = -100
best_model = None

measure = 'R2'
flag_pct = False

learning_rate = 0.001
sequence_size = [100, 150]
nr_hidden_units = [32, 64, 88]
max_iter = [500, 600, 400, 200]
episode_values = [max_iter[0]]
for el in max_iter[1:]:
    episode_values.append(episode_values[-1]+el)

nCols = len(sequence_size)
_, axs = subplots(1, nCols, figsize=(nCols*HEIGHT, HEIGHT), squeeze=False)
values = {}
for s in range(len(sequence_size)):
    length = sequence_size[s]
    trnX, trnY = sliding_window(train, seq_length=length)
    trnY = trnY[:, 1]
    trnX, trnY = Variable(Tensor(trnX)), Variable(Tensor(trnY))
    tstX, tstY = sliding_window(test, seq_length=length)
    tstY = tstY[:, 1]
    tstX, tstY = Variable(Tensor(tstX)), Variable(Tensor(tstY))

    for k in range(len(nr_hidden_units)):
        hidden_units = nr_hidden_units[k]
        yvalues = []
        model = DS_LSTM(input_size=nr_features,
                        hidden_size=hidden_units, learning_rate=learning_rate)
        next_episode_i = 0
        for n in range(1, episode_values[-1]+1):
            model.fit(trnX, trnY)
            if n == episode_values[next_episode_i]:
                next_episode_i += 1
                prd_tst = model.predict(tstX)
                yvalues.append((PREDICTION_MEASURES[measure])(tstY, prd_tst))
                print(
                    (f'LSTM - seq length={length} hidden_units={hidden_units} and nr_episodes={n}->{yvalues[-1]:.2f}'))
                if yvalues[-1] > last_best:
                    best = (length, hidden_units, n)
                    last_best = yvalues[-1]
                    best_model = model
        values[hidden_units] = yvalues

    multiple_line_chart(
        episode_values, values, ax=axs[0, s], title=f'LSTM seq length={length}', xlabel='nr episodes', ylabel=measure, percentage=flag_pct)
print(
    f'Best results with seq length={best[0]} hidden={best[1]} episodes={best[2]} ==> measure={last_best:.2f}')
savefig(f'images/lstm/{file_tag}_lstm_study.png')
show()
#%%Best results

trnX, trnY = sliding_window(train, seq_length=best[0])
trnY = trnY[:, 1]
trainY = DataFrame(trnY)
trainY.index = train.index[best[0]+1:]
trainY.columns = [target]
trnX, trnY = Variable(Tensor(trnX)), Variable(Tensor(trnY))
prd_trn = best_model.predict(trnX)
prd_trn = DataFrame(prd_trn)
prd_trn.index = train.index[best[0]+1:]
prd_trn.columns = [target]

tstX, tstY = sliding_window(test, seq_length=best[0])
tstY = tstY[:, 1]
testY = DataFrame(tstY)
testY.index = test.index[best[0]+1:]
testY.columns = [target]
tstX, tstY = Variable(Tensor(tstX)), Variable(Tensor(tstY))
prd_tst = best_model.predict(tstX)
prd_tst = DataFrame(prd_tst)
prd_tst.index = test.index[best[0]+1:]
prd_tst.columns = [target]

plot_evaluation_results(trnY.data.numpy(), prd_trn, tstY.data.numpy(
), prd_tst, f'images/lstm/{file_tag}_lstm_eval.png')
show()
index_col= 'Date'
plot_forecasting_series(trainY, testY, prd_trn.values, prd_tst.values,
                        f'images/lstm/{file_tag}_lstm_plots.png', x_label=index_col, y_label=target)
show()

# %%
