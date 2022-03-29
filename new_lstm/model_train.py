import os
import myModule
from myModule import Sequence
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import sys

to_screen = sys.stdout

# load data
df_train = pd.read_csv('log_11_2022-3-22-16-43-18_normal_data_last.csv')
y_index = 0
X_training, y_training = myModule.preprocess(df_train, y_index)
X_training = myModule.normalize(X_training)
y_training = myModule.normalize(y_training)

# sliding window length
stride = 5

X_training, y_training = myModule.reconstruct_data(X_training, y_training, stride)

# cut off the landing part
X_training = X_training[0:8000]
y_training = y_training[0:8000]

# split the training and validation dataset
X_training, X_valid, y_training, y_valid = train_test_split(X_training, y_training,
                                                            test_size=0.2, shuffle=False, random_state=0)

# create the output directory
parent_path = os.getcwd()
directory = '3layers2dropout20'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sys.stdout = open(output_dir + '/output.txt', 'a')
print("X_training.shape", X_training.shape)
print("y_training.shape", y_training.shape)
print("X_valid.shape", X_valid.shape)
print("y_valid.shape", np.array(y_valid).shape)
print("sliding window length", stride)
sys.stdout.close()
sys.stdout = to_screen


model_path = output_dir + '/model.pth'
threshold_file = output_dir + '/threshold.pickle'

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

sys.stdout = open(output_dir + '/output.txt', 'a')
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

# compute validation accuracy
err_output = []
valid_y_prediction = []
for i in range(0, len(X_valid)):
    inpt = [X_valid[i]]
    target = y_valid[i]

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
myModule.two_line('training result', x_label, y_label, train_y_prediction, y_training, 'training result', output_dir)

# validation result
myModule.two_line('validation result', x_label, y_label, valid_y_prediction, y_valid, 'validation result', output_dir)

print('finish')


