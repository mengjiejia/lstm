import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


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


b = 0.1  # Decay between samples (in (0, 1)).


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
        if filter_loss[i] >= threshold:
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

    #TPR = TP / (TP + FN)
    #FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print("TP", TP)
    print("FP", FP)
    print("TN", TN)
    print("FN", FN)

    # return TPR, FPR, ACC, FD
    return ACC, FD


# def process_data(X_train, y_train, X_val, y_val, stride):

class Sequence(torch.nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = torch.nn.LSTMCell(11, 128)
        self.drop_out1 = torch.nn.Dropout(0.2)
        self.lstm2 = torch.nn.LSTMCell(128, 64)
        self.drop_out2 = torch.nn.Dropout(0.2)
        self.lstm3 = torch.nn.LSTMCell(64, 32)
        #self.drop_out3 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(32,16)
        self.fc2 = torch.nn.Linear(16, 1)

        self.device = torch.device('cuda')

    def custom_loss_func(y_predictions, target):
        square_difference = torch.square(y_predictions - target)
        loss_value = torch.sum(square_difference) * 0.5
        return loss_value

    def forward(self, input):
        # Initial cell states
        h_t1 = torch.zeros(input.size(1), 128, dtype=torch.float32).to(self.device)
        c_t1 = torch.zeros(input.size(1), 128, dtype=torch.float32).to(self.device)
        h_t2 = torch.zeros(input.size(1), 64, dtype=torch.float32).to(self.device)
        c_t2 = torch.zeros(input.size(1), 64, dtype=torch.float32).to(self.device)
        h_t3 = torch.zeros(input.size(1), 32, dtype=torch.float32).to(self.device)
        c_t3 = torch.zeros(input.size(1), 32, dtype=torch.float32).to(self.device)

        outputs = []
        batch_size = input.size(1)
        seq_length = input.size(0)
        input = input.view(seq_length, batch_size, -1)
        for i in range(input.size(0)):
            # print(f" step {i} , Input : {input[i].shape}")
            # print(f" step {i} , hidden state 1: {h_t1.shape}")
            # print(f" step {i} , cell state 1: {c_t1.shape}")
            # print(f" step {i} , hidden state 2: {h_t2.shape}")
            # print(f" step {i} , cell state 2: {c_t2.shape}")
            # print(f" step {i} , hidden state 3: {h_t3.shape}")
            # print(f" step {i} , cell state 3: {c_t3.shape}")
            h_t1, c_t1 = self.lstm1(input[i], (h_t1, c_t1))
            # print(f" step {i} , hidden state 1: {h_t1.shape}")
            # print(f" step {i} , cell state 1: {c_t1.shape}")
            h_t1 = self.drop_out1(h_t1)
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t2 = self.drop_out2(h_t2)
            # print(f" step {i} , hidden state 2: {h_t2.shape}")
            # print(f" step {i} , cell state 2: {c_t2.shape}")
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.fc1(h_t3)
            output = self.fc2(output)
            #print('output i', output)

        # outputs.append(output)
        # print("outputs.shape", np.array(outputs).shape)
        # print(outputs)
        #print('return output', output)
        # output = self.fc1(h_t2)
        # output = self.fc2(output)
        return output


# import dataset

#filename = ''
#read_path = '/home/mengjie/dataset/px4/multicopter/normal/training_data/' + filename
#df_train1 = pd.read_csv('log_0_2022-3-14-14-40-35_normal_data_last.csv')
#df_train2 = pd.read_csv('log_0_2022-3-14-14-49-48_normal_data_last.csv')
#df_train3 = pd.read_csv('log_8_2022-3-12-17-48-52_normal_data_last.csv')
#df_noturn = pd.read_csv('log_3_2022-3-18-16-22-16_normal_data_last.csv')
# df_fixed_train = pd.read_csv('log_10_2022-3-22-12-11-27_normal_data_last.csv')
df_fixed_train = pd.read_csv('log_11_2022-3-22-16-43-18_normal_data_last.csv')


# df_train4 = pd.read_csv('log_15_2022-3-14-14-35-00_normal_data_last.csv')

# dfs = [df_train1, df_train2, df_train3]
dfs = [df_fixed_train]

# df_train = pd.read_csv('rate_simulation_normal_data_last_312.csv')
# df_valid = pd.read_csv('log_13_2022-3-14-14-04-09_normal_data_last.csv')
#df_valid = pd.read_csv('log_10_2022-3-22-15-17-08_normal_data_last.csv')
df_valid = pd.read_csv('log_15_2022-3-22-17-06-08_normal_data_last.csv')


# df_test = pd.read_csv('log_15_2022-3-14-14-35-00_normal_data_last.csv')
#df_test = pd.read_csv('log_8_2022-3-22-15-11-05_normal_data_last.csv')
df_test = pd.read_csv('log_8_2022-3-23-11-40-47_normal_data_last.csv')
#df_test = pd.read_csv('log_11_2022-3-22-16-43-18_normal_data_last.csv')

# drop out the monitored datasource
# X_tr = df_train.loc[df_train['datasource'] != 'roll_measured']
# train_dataset = df_train.drop(columns=['datasource'])
# train_dataset = train_dataset.to_numpy()
df_valid.rename(columns={'Unnamed: 0': 'datasource'}, inplace=True)
df_test.rename(columns={'Unnamed: 0': 'datasource'}, inplace=True)

valid_dataset = df_valid.drop(columns=['datasource'])
valid_dataset = valid_dataset.to_numpy()
# valid_dataset = np.delete(valid_dataset, 6, 0)

test_dataset = df_test.drop(columns=['datasource'])
test_dataset = test_dataset.to_numpy()
# test_dataset = np.delete(test_dataset, 6, 0)

# split the dataset into training and validation, the first 80% is for training

# train_pct_index = int(0.8 * len(train_dataset[0]))

# the predicted data source index
y_index = 0
# y = train_dataset[y_index]
y_valid = valid_dataset[y_index]
y_test = test_dataset[y_index]

# train_dataset = train_dataset.T
valid_dataset = valid_dataset.T
test_dataset = test_dataset.T

# X_train, X_valid = train_dataset[:train_pct_index, :], train_dataset[train_pct_index:, :]
#
# y_train, y_valid = y[:train_pct_index], y[train_pct_index:]
X_valid = valid_dataset
X_test = test_dataset
# normalize data
# X_train = normalize(X_train)
# y_train = normalize(y_train)

X_valid = normalize(X_valid)
y_valid = normalize(y_valid)
print('y_valid min', min(y_valid[11:3012]))
print('y_valid max', max(y_valid[11:3012]))

X_test = normalize(X_test)
y_test = normalize(y_test)
print('y_test min', min(y_test[11:10012]))
print('y_test max', max(y_test[11:10012]))

# X_test = normalize(X_test)
# y_test = normalize(y_test)

# sliding window length
stride = 5

# X_train_input, y_train_input = process_data(X_train, y_train, stride)
X_valid_input, y_valid_input = process_data(X_valid, y_valid, stride)
X_test_input, y_test_input = process_data(X_test, y_test, stride)
# X_test_input, y_test_input = process_data(X_test, y_test, stride)

# X_train_input = np.array(X_train_input)
X_valid_input = np.array(X_valid_input)
X_test_input = np.array(X_test_input)
# X_test_input = np.array(X_train_input)

X_training = []
y_training = []
for df in dfs:
    df.rename(columns={'Unnamed: 0': 'datasource'}, inplace=True)
    df = df.drop(columns=['datasource'])
    # print(df.shape)
    df = df.to_numpy()
    # df = np.delete(df, 6, 0)
    y = normalize(df[y_index])
    print('y_train min', min(y[11:8012]))
    print('y_train max', max(y[11:8012]))
    X = normalize(df.T)
    X_input, y_input = process_data(X, y, stride)
    for i in range(len(X_input)):
        X_training.append(X_input[i])
        y_training.append(y_input[i])

X_training = np.array(X_training)
y_training = np.array(y_training)

# cut the landing part
X_training = X_training[0:8000]
y_training = y_training[0:8000]
X_training, X_valid_input, y_training, y_valid_input = train_test_split(X_training, y_training,
    test_size=0.2, shuffle = False, random_state = 0)
# X_valid_input = X_valid_input[0:3000]
# y_valid_input = y_valid_input[0:3000]
X_test_input = X_test_input[0:10000]
y_test_input = y_test_input[0:10000]
#3500

print("X_training.shape", X_training.shape)
print("y_training.shape", y_training.shape)
print("X_valid.shape", X_valid_input.shape)
print("y_valid.shape", np.array(y_valid_input).shape)
print("X_test.shape", X_test_input.shape)
print("y_test.shape", np.array(y_test_input).shape)


# inject abnormal data
test_label = [0] * len(y_test_input)
for i in range(2000, 4000):
    for j in range(stride):  # 3 is the roll_rate index
        #for k in range(11):
        X_test_input[i][j][4] = X_test_input[i][j][4]* 2;
    y_test_input[i] = y_test_input[i] * 2

    test_label[i] = 1

# print("X_train_input shape:", X_train_input.shape)
# print("X_valid_input shape:", X_valid_input.shape)
# print("X_test_input shape:", X_test_input.shape)

# number of data sources
n_features = 11
# this is length of sliding window.

n_steps = stride
batch_size = 16
#PATH = 'TEST.pth'
PATH = 'test_samenorandom_cut_fixed_wing_custom_loss_sequence_training_model.pth'
test_path = PATH
# criterion = torch.nn.L1Loss()  # reduction='sum' created huge loss value
# criterion = Sequence.custom_loss_func()

train = False

if train == True:
    # create lstm
    mv_lstm = Sequence()
    mv_lstm.float()
    mv_lstm = mv_lstm.cuda()
    # criterion = torch.nn.L1Loss()  # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)

    train_episodes = 500

    # training model
    min_valid_loss = np.inf

    epoch_train_loss = []
    epoch_valid_loss = []

    for t in range(train_episodes):
        mv_lstm.train()
        train_loss = []
        for i in range(0, len(X_training), batch_size):
            inpt = X_training[i:i + batch_size, :, :]
            target = y_training[i:i + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)

            x_batch = x_batch.permute(1, 0, 2)
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            optimizer.zero_grad()
            output = mv_lstm(x_batch)
            loss = Sequence.custom_loss_func(output.view(-1), y_batch)
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in mv_lstm.parameters())

            loss = loss + l2_lambda * l2_norm

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

            output = mv_lstm(x_valid)

            loss = Sequence.custom_loss_func(output.view(-1), y_target)

            valid_loss.append(loss.item())

        epoch_valid_loss.append(np.mean(valid_loss))

        print(
            f'Epoch {t} \t\t Training Loss: {np.mean(train_loss)} \t\t Validation Loss: {np.mean(valid_loss)}')

    # Saving model
    torch.save(mv_lstm, PATH)

    test_ = plt.plot(epoch_valid_loss)

    plt.savefig('test_sequence_df2_loss_last.png')
    plt.clf()

    train_ = plt.plot(epoch_train_loss)

    plt.savefig('test_sequence_df2_training_loss_last.png')

print("computing threshold")

# Load
my_lstm = torch.load(PATH)
print('lstm parameters')
for name, m in my_lstm.named_modules():
    print(name)
    print(m)
# for item in my_lstm.parameters():
#     print(item)
params = my_lstm.state_dict()
print(params.keys())
for k, v in params.items():
    print(k)
    print(v)
    print(v.size())


my_lstm.eval()
Filtered = []
train_y_predicted = []
err_abs = torch.nn.L1Loss()
for i in range(0, len(X_training)):
    inpt = [X_training[i, :, :]]
    target = y_training[i]

    x_batch = torch.tensor(inpt, dtype=torch.float32)
    y_batch = torch.tensor(target, dtype=torch.float32)
    x_batch = x_batch.permute(1, 0, 2)
    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()

    output = my_lstm(x_batch)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        train_y_predicted.append(prediction[k])

    err = err_abs(output.view(-1), y_batch)
    err = torch.mean(err)
    err = err.to("cpu").detach().numpy()

    if i == 0:
        low_pass_IIR = LowPassIIR(b, err)

    filtered_err = low_pass_IIR.filter(err.item())
    Filtered.append(filtered_err.item())
    # print(filtered_loss)
    # if filtered_loss < 1:
    #     Filtered.append(filtered_loss)


filter_mean = np.mean(Filtered)
filter_std = np.std(Filtered)
FD_threshold = filter_mean + 2.6 * filter_std
#FD_threshold = 0.05902386768298962
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

err_output = []
valid_y_predicted = []
for i in range(0, len(X_valid_input)):
    inpt = [X_valid_input[i]]
    target = y_valid_input[i]

    x_valid = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_valid = x_valid.cuda()
    y_target = y_target.cuda()

    x_valid = x_valid.permute(1, 0, 2)
    output = my_lstm(x_valid)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        valid_y_predicted.append(prediction[k])

    err = err_abs(y_target, output.view(-1))
    err = torch.mean(err)
    err = err.to("cpu").detach().numpy()

    # if err > 1.0:
    #     print("err", i)
    #     print(err)

    if i == 0:
        low_pass_IIR = LowPassIIR(b, err.item())

    filtered_err = low_pass_IIR.filter(err.item())
    err_output.append(filtered_err)

total = len(err_output)
normal_count = 0
abnormal_count = 0
for item in err_output:

    if item <= FD_threshold:
        normal_count += 1
    else:
        abnormal_count += 1
accuracy = normal_count / total
print("valid accuracy", accuracy)

# test
test_err = []
test_y_predicted = []

for i in range(0, len(X_test_input)):
    inpt = [X_test_input[i]]
    target = y_test_input[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        test_y_predicted.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = LowPassIIR(b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    test_err.append(test_filtered_loss)

ACC, FD = Accuracy(test_label, test_err, FD_threshold)
# print("TPR", TPR)
# print("FPR", FPR)
print("ACC", ACC)

# print("train prediction", train_y_predicted)
# print("valid prediction", valid_y_predicted)
# print("test prediction", test_y_predicted)

plt.figure(figsize=(12,6))
plt.title("train")
plt.xlabel('timestamp')
plt.ylabel('roll_angle')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(len(train_y_predicted)), train_y_predicted, 'r', linewidth=2.0, label='prediction')
plt.plot(np.arange(len(train_y_predicted)), y_training, 'b', linewidth=2.0, label='groundtruth')
plt.legend()
plt.savefig('test_whole_fixed_wing_new_train_prediction.png')
plt.clf()

plt.figure(figsize=(12,6))
plt.title("valid")
plt.xlabel('timestamp')
plt.ylabel('roll_angle')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(len(valid_y_predicted)), valid_y_predicted, 'r', linewidth=2.0, label='prediction')
plt.plot(np.arange(len(valid_y_predicted)), y_valid_input, 'b', linewidth=2.0, label='groundtruth')
plt.legend()
plt.savefig('test_whole_fixed_wing_new_valid_prediction.png')
plt.clf()

plt.figure(figsize=(12,6))
plt.title("test")
plt.xlabel('timestamp')
plt.ylabel('roll_angle')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(len(test_y_predicted)), test_y_predicted, 'r', linewidth=2.0, label='prediction')
plt.plot(np.arange(len(test_y_predicted)), y_test_input, 'b', linewidth=2.0, label='groundtruth')
plt.legend()
plt.title('attack yaw rate [2000, 4000], threshold ' + str(FD_threshold))
plt.savefig('test_whole_fixed_wing_roll_new_test_prediction_15.png')
plt.clf()

plt.figure(figsize=(12,6))
plt.title("test")
plt.xlabel('timestamp')
plt.ylabel('error')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(len(test_err)), test_err, 'r', linewidth=2.0, label='test_err')
plt.legend()
plt.title('test_err roll angle [2000, 4000], threshold ' + str(FD_threshold))
plt.savefig('test_whole_err_fixed_wing_roll_new_test_prediction.png')
plt.clf()

#
#
