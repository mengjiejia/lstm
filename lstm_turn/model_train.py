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
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

to_screen = sys.stdout

# create the output directory
parent_path = os.getcwd()
directory = 'sequence/range_scale/normal_training/turn3'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load data
filename = 'log_1_2022-5-10-18-07-45_normal_data_last_degrees.csv'
read_path = '/home/mengjie/dataset/px4/fixed_wing/turn/3/'
df_train1 = pd.read_csv(read_path + filename)  # turn cut 800 - 2500

y_index = 0
X_training, y_training = myModule.preprocess(df_train1, y_index)

# cut off the landing part
X_training = X_training[800:2500]
y_training = y_training[800:2500]

# scale the data with different normalization
# X_training = np.delete(X_training, y_index, 1)
#
# y_training = y_training.reshape((len(y_training), 1))


# min_max scaler
# X_scaler = MinMaxScaler()
# X_scaler.fit(X_training)
# X_training = X_scaler.transform(X_training)

# scale y in range[-1, 1], divided by 90 degree
# y_training = y_training / 45
# X_training = np.insert(X_training, y_index, y_training, axis=1)

df_x = pd.DataFrame(X_training)
df_y = pd.DataFrame(y_training)

df_x.to_csv(output_dir + '/df_x.csv')
df_y.to_csv(output_dir + '/df_y.csv')

# y_scaler = MinMaxScaler()
# y_scaler.fit(y_training)
# y_training = y_scaler.transform(y_training)

# sliding window length
stride = 5
# y_training = y_training.reshape(len(y_training), )

X_training, y_training = myModule.reconstruct_data(X_training, y_training, stride)

X_training_normal = X_training
y_training_normal = y_training

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
pattern = -1
percentage = 0.2
if pattern != -1:
    X_training_abnormal = myModule.abnormal_injection(X_training_abnormal, pattern, percentage, y_index)
    X_valid_abnormal = myModule.abnormal_injection(X_valid_abnormal, pattern, percentage, y_index)

# X_training_combined = np.concatenate((X_training_normal, X_training_abnormal))
# y_training_combined = np.concatenate((y_training_normal, y_training_abnormal))
#
# X_valid_combined = np.concatenate((X_valid_normal, X_valid_abnormal))
# y_valid_combined = np.concatenate((y_valid_normal, y_valid_abnormal))

# X_training_combined = np.array(X_training_combined)
# y_training_combined = np.array(y_training_combined)

# X_valid_combined = np.array(X_valid_combined)
# y_valid_combined = np.array(y_valid_combined )


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
threshold_file_abs = output_dir + '/threshold_abs.pickle'
mean_std_file_abs = output_dir + '/mean_std_abs.pickle'
threshold_file_per = output_dir + '/threshold_per.pickle'
mean_std_file_per = output_dir + '/mean_std_per.pickle'

# dump(X_scaler, output_dir + '/X_scaler.joblib')
# dump(y_scaler, output_dir + '/y_scaler.joblib')

# number of data sources
n_features = 10  # delete altitude
# this is length of sliding window.

n_steps = stride
batch_size = 16

# create lstm
mv_lstm = myModule.Sequence()
mv_lstm.float()
mv_lstm = mv_lstm.cuda()
optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)

train_episodes = 300

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

        loss = myModule.custom_loss_func(output.view(-1), y_batch)
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
Filtered_err_percentage = []
low_pass_IIR = 0
percentage_low_pass_IIR = 0
# regression evaluation metrics
# mse = torch.mse

# when computing threshold, feed the training data one by one, instead of by batch
# y_training_normal = y_scaler.inverse_transform(y_training_normal)
y_training_normal = y_training_normal * 90
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
    prediction = prediction * 90
    # prediction = prediction.reshape((len(prediction), 1))
    # prediction = y_scaler.inverse_transform(prediction)
    # prediction = prediction.reshape((len(prediction), ))

    for k in range(len(prediction)):
        train_y_prediction.append(prediction[k])

    # err = err_abs(output.view(-1), y_batch)
    # err = torch.mean(err)
    # err = err.to("cpu").detach().numpy()
    err = abs(prediction - target)
    err_percentage = err.item() / abs(target)

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err)
        percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage)

    filtered_err = low_pass_IIR.filter(err.item())
    Filtered.append(filtered_err.item())

    filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
    Filtered_err_percentage.append(filtered_err_percentage.item())

print("computing threshold")

filter_mean_abs = np.mean(Filtered)
filter_std_abs = np.std(Filtered)
FD_threshold_abs = filter_mean_abs + 2 * filter_std_abs

filter_mean_per = np.mean(Filtered_err_percentage)
filter_std_per = np.std(Filtered_err_percentage)
FD_threshold_per = filter_mean_per + 2 * filter_std_per

with open(threshold_file_per, 'wb') as f:
    pickle.dump(FD_threshold_per, f)

with open(mean_std_file_per, 'wb') as f:
    pickle.dump([filter_mean_per, filter_std_per], f)

with open(threshold_file_abs, 'wb') as f:
    pickle.dump(FD_threshold_abs, f)

with open(mean_std_file_abs, 'wb') as f:
    pickle.dump([filter_mean_abs, filter_std_abs], f)

sys.stdout = open(output_dir + '/output.txt', 'a')
print("percentage error mean", filter_mean_per)
print("percentage error std", filter_std_per)
print("percentage FD_threshold", FD_threshold_per)

print("error mean", filter_mean_abs)
print("error std", filter_std_abs)
print("FD_threshold", FD_threshold_abs)

total = len(Filtered_err_percentage)
normal_count = 0
abnormal_count = 0
for item in Filtered_err_percentage:
    if item <= FD_threshold_per:
        normal_count += 1
    else:
        abnormal_count += 1
accuracy = normal_count / total
print("train accuracy", accuracy)

# compute validation accuracy
err_output = []
valid_err_percentage = []
valid_y_prediction = []
# y_valid_normal = y_scaler.inverse_transform(y_valid_normal)
y_valid_normal = y_valid_normal * 90
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
    prediction = prediction * 90
    # prediction = prediction.reshape((len(prediction), 1))
    # prediction = y_scaler.inverse_transform(prediction)
    # prediction = prediction.reshape((len(prediction), ))
    for k in range(len(prediction)):
        valid_y_prediction.append(prediction[k])

    err = abs(prediction - target)
    err_percentage = err.item() / abs(target)

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())
        percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage.item())

    filtered_err = low_pass_IIR.filter(err.item())
    err_output.append(filtered_err)

    filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
    valid_err_percentage.append(filtered_err_percentage)

total = len(err_output)
normal_count = 0
abnormal_count = 0
for item in valid_err_percentage:

    if item <= FD_threshold_per:
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
myModule.two_line('training result', x_label, y_label, train_y_prediction, y_training_normal,
                  'training result',
                  output_dir)

# validation result
myModule.two_line('validation result', x_label, y_label, valid_y_prediction, y_valid_normal,
                  'validation result',
                  output_dir)

# test error
myModule.one_line('training error', x_label, y_label, Filtered, 'training error', output_dir)

print('finish')
