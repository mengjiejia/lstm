import os
import myModule
from myModule import Sequence
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import sys
import random

to_screen = sys.stdout

# load data
df_train1 = pd.read_csv('log_11_2022-3-22-16-43-18_normal_data_last.csv')  # cut 8000
df_train2 = pd.read_csv('log_15_2022-3-22-17-06-08_normal_data_last.csv')  # cut 1000 - 3000
df_train3 = pd.read_csv('log_21_2022-3-31-14-40-08_normal_data_last.csv')  # cut 1000 - 4000

y_index = 0
X_training, y_training = myModule.preprocess(df_train1, y_index)

# cut off the landing part
# X_training = X_training[1000:4000]
# y_training = y_training[1000:4000]

X_training = myModule.normalize(X_training)
y_training = myModule.normalize(y_training)

# sliding window length
stride = 5

X_training, y_training = myModule.reconstruct_data(X_training, y_training, stride)

# cut off the landing part
X_training_normal = X_training[0:8000]
y_training_normal = y_training[0:8000]

X_training_abnormal = X_training_normal.copy()
y_training_abnormal = y_training_normal.copy()

# split the training and validation dataset
X_training_normal, X_valid_normal, y_training_normal, y_valid_normal = train_test_split(X_training_normal,
                                                                                        y_training_normal,
                                                                                        test_size=0.2, shuffle=False,
                                                                                        random_state=0)

X_training_abnormal, X_valid_abnormal, y_training_abnormal, y_valid_abnormal = train_test_split(X_training_abnormal,
                                                                                                y_training_abnormal,
                                                                                                test_size=0.2,
                                                                                                shuffle=False,
                                                                                                random_state=0)

# random shuffle
# training
# X_training, y_training = myModule.random_shuffle(X_training_normal, y_training_normal)
#
# # validation
# X_valid, y_valid = myModule.random_shuffle(X_valid_normal, y_valid_normal)


# inject abnormal data
pattern = 1
percentage = 0.2
rate = 0.5
if pattern != -1:
    X_training_abnormal = myModule.abnormal_injection(X_training_abnormal, pattern, percentage, y_index, rate)
    X_valid_abnormal = myModule.abnormal_injection(X_valid_abnormal, pattern, percentage, y_index, rate)

# X_training_combined = np.concatenate((X_training_normal, X_training_abnormal))
# y_training_combined = np.concatenate((y_training_normal, y_training_abnormal))
#
# X_valid_combined = np.concatenate((X_valid_normal, X_valid_abnormal))
# y_valid_combined = np.concatenate((y_valid_normal, y_valid_abnormal))

# X_training_combined = np.array(X_training_combined)
# y_training_combined = np.array(y_training_combined)

# X_valid_combined = np.array(X_valid_combined)
# y_valid_combined = np.array(y_valid_combined )

# create the output directory
parent_path = os.getcwd()
directory = 'test_abnormal_training_decrease'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if pattern == 0:
    abnormal_pattern = 'continuously'
elif pattern == 1:
    abnormal_pattern = 'random'
elif pattern == -1:
    abnormal_pattern = 'normal'

sys.stdout = open(output_dir + '/output.txt', 'a')
print(f"######### training output {abnormal_pattern} ##########")
# print("X_training_combined.shape", X_training_combined.shape)
# print("y_training_combined.shape", y_training_combined.shape)
# print("X_training_normal.shape", X_training_normal.shape)
# print("y_training_normal.shape", y_training_normal.shape)
# print("X_training_abnormal.shape", X_training_abnormal.shape)
# print("y_training_abnormal.shape", y_training_abnormal.shape)
print("X_training.shape", X_training_abnormal.shape)
print("y_training.shape", y_training_abnormal.shape)
# print('training abnormal windows', int(len(X_training_abnormal) * percentage))
print("X_valid.shape", X_valid_abnormal.shape)
print("y_valid.shape", y_valid_abnormal.shape)
# print('valid abnormal windows', int(len(X_valid_abnormal) * percentage))
print("sliding window length", stride)
sys.stdout.close()
sys.stdout = to_screen

model_path = output_dir + '/model.pth'
threshold_file = output_dir + '/threshold.pickle'
mean_std_file = output_dir + '/mean_std.pickle'

# number of data sources
n_features = 11
# this is length of sliding window.

n_steps = stride
batch_size = 16

# create lstm
mv_lstm = myModule.Sequence()
mv_lstm.float()
mv_lstm = mv_lstm.cuda()
optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)

train_episodes = 500

# training model
min_valid_loss = np.inf
epoch_train_loss = []

for t in range(train_episodes):
    mv_lstm.train()
    train_loss = []
    for i in range(0, len(X_training_abnormal), batch_size):
        inpt = X_training_abnormal[i:i + batch_size, :, :]
        target = y_training_normal[i:i + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)

        x_batch = x_batch.permute(1, 0, 2)
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        optimizer.zero_grad()
        output = mv_lstm(x_batch)
        loss = myModule.Sequence.custom_loss_func(output.view(-1), y_batch)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                      for p in mv_lstm.parameters())

        loss = loss + l2_lambda * l2_norm

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    epoch_train_loss.append(np.mean(train_loss))
    print(
        f'Epoch {t} \t\t Training Loss: {np.mean(train_loss)} ')

# Saving model
torch.save(mv_lstm, model_path)

train_ = plt.plot(epoch_train_loss)
plt.savefig(output_dir + '/training_loss.png')

# compute training accuracy and threshold
my_lstm = torch.load(model_path)
my_lstm.eval()
Filtered = []
train_y_prediction = []
err_abs = torch.nn.L1Loss()

# regression evaluation metrics
# mse = torch.mse


for i in range(0, len(X_training_abnormal)):
    inpt = [X_training_abnormal[i, :, :]]
    target = y_training_normal[i]

    x_batch = torch.tensor(inpt, dtype=torch.float32)
    y_batch = torch.tensor(target, dtype=torch.float32)
    x_batch = x_batch.permute(1, 0, 2)
    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()

    output = my_lstm(x_batch)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        train_y_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_batch)
    err = torch.mean(err)
    err = err.to("cpu").detach().numpy()

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err)

    filtered_err = low_pass_IIR.filter(err.item())
    Filtered.append(filtered_err.item())

print("computing threshold")
filter_mean = np.mean(Filtered)
filter_std = np.std(Filtered)
FD_threshold = filter_mean + 2 * filter_std

with open(threshold_file, 'wb') as f:
    pickle.dump(FD_threshold, f)

with open(mean_std_file, 'wb') as f:
    pickle.dump([filter_mean, filter_std], f)

sys.stdout = open(output_dir + '/output.txt', 'a')
print("error mean", filter_mean)
print("error std", filter_std)
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

# compute validation accuracy
err_output = []
valid_y_prediction = []
for i in range(0, len(X_valid_abnormal)):
    inpt = [X_valid_abnormal[i]]
    target = y_valid_normal[i]

    x_valid = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_valid = x_valid.cuda()
    y_target = y_target.cuda()

    x_valid = x_valid.permute(1, 0, 2)
    output = my_lstm(x_valid)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        valid_y_prediction.append(prediction[k])

    err = err_abs(y_target, output.view(-1))
    err = torch.mean(err)
    err = err.to("cpu").detach().numpy()

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

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

sys.stdout.close()
sys.stdout = to_screen

x_label = 'sliding window number'
y_label = 'roll angle'

# training result
myModule.two_line('training result', x_label, y_label, train_y_prediction[0:100], y_training_normal[0:100], 'training result',
                  output_dir)

# validation result
myModule.two_line('validation result', x_label, y_label, valid_y_prediction[0:100], y_valid_normal[0:100], 'validation result',
                  output_dir)

# test error
myModule.one_line('training error', x_label, y_label, Filtered, 'training error', output_dir)

print('finish')
