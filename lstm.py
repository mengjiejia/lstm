import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


# multivariate data preparation

# split a multivariate sequence into samples

def process_data(X_train, y_train, stride):
    # inout_seq = []
    X = []
    y = []
    L = len(X_train)
    for i in range(L - stride):
        train_x = X_train[i:i + stride, :]
        train_y = y_train[i + stride:i + stride + 1]
        X.append(train_x)
        y.append(train_y)
        # train_x = torch.tensor(train_x, dtype=torch.float32)
        # train_y = torch.tensor(train_y, dtype=torch.float32)
        # inout_seq.append((train_x, train_y))
    return X, y


# normalize data using zscore
def normalize(data):
    data = zscore(data, axis=0)
    data = np.nan_to_num(data)
    # data = torch.tensor(data, dtype=torch.float32)
    return data


b = 0.9  # Decay between samples (in (0, 1)).


class LowPassIIR:
    def __init__(self, b, x0):
        self.a = 1 - b
        self.reset(x0)

    def reset(self, x0):
        self.y = x0

    def filter(self, x):
        self.y += self.a * (x - self.y)
        return self.y


def Accuracy(label, filter_loss, threshold):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    FD = []
    for i in range(len(filter_loss)):
        # abnormal, positive
        if filter_loss[i] > threshold:
            FD.append(1)
            if label[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            # normal, negative
            FD.append(0)
            if label[i] == 0:
                TN += 1
            else:
                FN += 1

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("TP", TP)
    print("FP", FP)
    print("TN", TN)
    print("FN", FN)

    return TPR, FPR, ACC, FD


# def process_data(X_train, y_train, X_val, y_val, stride):


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 128  # number of hidden states
        self.n_layers = 3  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)
        self.device = torch.device('cuda')

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


# import dataset
df_train = pd.read_csv('simulation_normal_data_last_312.csv')
df_test = pd.read_csv('test_data_normal_last3.csv')

# drop out the monitored datasource
# X_tr = df_train.loc[df_train['datasource'] != 'roll_measured']
train_dataset = df_train.drop(columns=['datasource'])
train_dataset = train_dataset.to_numpy()

test_dataset = df_test.drop(columns=['datasource'])
test_dataset = test_dataset.to_numpy()

# split the dataset into training and validation, the first 80% is for training

train_pct_index = int(0.8 * len(train_dataset[0]))

# the predicted data source index
y_index = 0
y = train_dataset[y_index]
y_test = test_dataset[y_index]

train_dataset = train_dataset.T
test_dataset = test_dataset.T

X_train, X_valid = train_dataset[:train_pct_index, :], train_dataset[train_pct_index:, :]

y_train, y_valid = y[:train_pct_index], y[train_pct_index:]
X_test = test_dataset
# normalize data
X_train = normalize(X_train)
y_train = normalize(y_train)

X_valid = normalize(X_valid)
y_valid = normalize(y_valid)

X_test = normalize(X_test)
y_test = normalize(y_test)

# sliding window length
stride = 15

X_train_input, y_train_input = process_data(X_train, y_train, stride)
X_valid_input, y_valid_input = process_data(X_valid, y_valid, stride)
X_test_input, y_test_input = process_data(X_test, y_test, stride)

X_train_input = np.array(X_train_input)
X_valid_input = np.array(X_valid_input)
X_test_input = np.array(X_train_input)

print("X_train_input shape:", X_train_input.shape)
print("X_valid_input shape:", X_valid_input.shape)
# print("X_test_input shape:", X_test_input.shape)

# number of data sources
n_features = 6
# this is length of sliding window
n_steps = stride

# create lstm
mv_lstm = MV_LSTM(n_features, n_steps)
mv_lstm = mv_lstm.cuda()
criterion = torch.nn.L1Loss()  # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)

train_episodes = 500

batch_size = 16

# training model
min_valid_loss = np.inf
PATH = 'saved_model_last_simulation.pth'

epoch_train_loss = []
epoch_valid_loss = []

for t in range(train_episodes):
    mv_lstm.train()
    train_loss = []
    for i in range(0, len(X_train_input), batch_size):
        inpt = X_train_input[i:i + batch_size, :, :]
        target = y_train_input[i:i + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        mv_lstm.init_hidden(x_batch.size(0))
        optimizer.zero_grad()
        output = mv_lstm(x_batch)
        loss = criterion(output.view(-1), y_batch)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    epoch_train_loss.append(np.mean(train_loss))

    mv_lstm.eval()
    valid_loss = []

    for j in range(0, len(X_valid_input)):
        inpt = [X_valid_input[j]]
        target = y_valid_input[j]

        x_valid = torch.tensor(inpt, dtype=torch.float32)
        y_target = torch.tensor(target, dtype=torch.float32)
        x_valid = x_valid.cuda()
        y_target = y_target.cuda()

        mv_lstm.init_hidden(x_valid.size(0))

        output = mv_lstm(x_valid)

        loss = criterion(output.view(-1), y_target)

        valid_loss.append(loss.item())

    epoch_valid_loss.append(np.mean(valid_loss))

    print(
        f'Epoch {t} \t\t Training Loss: {np.mean(train_loss)} \t\t Validation Loss: {np.mean(valid_loss)}')

# Saving model
torch.save(mv_lstm, PATH)

test_ = plt.plot(epoch_valid_loss)

plt.savefig('30_validate_loss_last.png')
plt.clf()

train_ = plt.plot(epoch_train_loss)

plt.savefig('30_training_loss_last.png')


print("computing threshold")

# Load
my_lstm = torch.load(PATH)
my_lstm.eval()
Filtered = []
for i in range(0, len(X_train_input)):
    inpt = [X_train_input[i, :, :]]
    target = y_train_input[i]

    x_batch = torch.tensor(inpt, dtype=torch.float32)
    y_batch = torch.tensor(target, dtype=torch.float32)

    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()

    my_lstm.init_hidden(x_batch.size(0))

    output = my_lstm(x_batch)
    loss = criterion(output.view(-1), y_batch)
    print(output.view(-1))
    if i == 0:
        low_pass_IIR = LowPassIIR(b, loss.item())

    filtered_loss = low_pass_IIR.filter(loss.item())
    #print(filtered_loss)
    if filtered_loss < 1:
        Filtered.append(filtered_loss)

filter_mean = np.mean(Filtered)
filter_std = np.std(Filtered)
FD_threshold = filter_mean + 2.6 * filter_std
print("loss mean", filter_mean)
print("loss std", filter_std)
print("FD_threshold", FD_threshold)
total = len(Filtered)
normal_count = 0
abnormal_count = 0
for item in Filtered:
    if item <= FD_threshold:
        normal_count += 1
    else:
        abnormal_count += 1
accuracy = normal_count / total
print("train accuracy", accuracy)


loss_output = []

for b in range(0, len(X_valid_input)):
    inpt = [X_valid_input[b]]
    target = y_valid_input[b]

    x_valid = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_valid = x_valid.cuda()
    y_target = y_target.cuda()

    my_lstm.init_hidden(x_valid.size(0))

    output = my_lstm(x_valid)

    loss = criterion(output.view(-1), y_target)

    if b == 0:
        low_pass_IIR = LowPassIIR(b, loss.item())

    filtered_loss = low_pass_IIR.filter(loss.item())
    loss_output.append(filtered_loss)

total = len(loss_output)
normal_count = 0
abnormal_count = 0
for item in loss_output:
    if item <= FD_threshold:
        normal_count += 1
    else:
        abnormal_count += 1
accuracy = normal_count / total
print("valid accuracy", accuracy)

print("roll_angle", y_train_input)
# loss_output = []

# for b in range(0, len(X_bias_input)):
#    inpt = [X_bias_input[b]]
#    target = y_valid_input[b]

#    x_bias = torch.tensor(inpt, dtype=torch.float32)
#    y_target = torch.tensor(target, dtype=torch.float32)
#    x_bias = x_bias.cuda()
#    y_target = y_target.cuda()

#    my_lstm.init_hidden(x_bias.size(0))

#    output = my_lstm(x_bias)

#    loss = criterion(output.view(-1), y_target)

#    if b == 0:
#        low_pass_IIR = LowPassIIR(b, loss.item())

#    filtered_loss = low_pass_IIR.filter(loss.item())
#    loss_output.append(filtered_loss)

# total = len(loss_output)
# normal_count = 0
# abnormal_count = 0
# for item in loss_output:
#    if item <= FD_threshold:
#        normal_count += 1
#    else:
#        abnormal_count += 1
# accuracy = normal_count / total
# print("valid_bias accuracy", accuracy)


# TPR, FPR, ACC, FD = Accuracy(label, loss_output, FD_threshold)

# print("TPR", TPR)
# print("FPR", FPR)
# print("ACC", ACC)


# loss_output = []

# for b in range(0, len(X_test_input)):
#    inpt = [X_test_input[b]]
#    target = y_test_input[b]

#    x_test= torch.tensor(inpt, dtype=torch.float32)
#    y_target = torch.tensor(target, dtype=torch.float32)
#    x_test = x_test.cuda()
#    y_target = y_target.cuda()

#    my_lstm.init_hidden(x_test.size(0))

#    output = my_lstm(x_test)

#    loss = criterion(output.view(-1), y_target)

#    if b == 0:
#        low_pass_IIR = LowPassIIR(b, loss.item())

#    filtered_loss = low_pass_IIR.filter(loss.item())
#    loss_output.append(filtered_loss)

# total = len(loss_output)
# normal_count = 0
# abnormal_count = 0
# for item in loss_output:
#    if item <= FD_threshold:
#        normal_count += 1
#    else:
#        abnormal_count += 1
# accuracy = normal_count / total
# print("test accuracy", accuracy)


# print('loss : ', loss.item())

# print(test_loss)
# test_ = plt.plot(test_loss)
#
# plt.savefig('validate_loss_last.png')
# plt.clf()
#
# train_ = plt.plot(train_loss)
#
# plt.savefig('training_loss_last.png')
