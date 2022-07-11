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

# number of data sources, delete altitude
n_features = 10

# predicted sensor
y_index = 0

# sliding window length
stride = 5

# model parameters
n_steps = stride
batch_size = 16
train_episodes = 300

# abnormal pattern in training dataset
pattern = -1

# abnormal percentage in training dataset
percentage = 0.2

to_screen = sys.stdout

# create the output directory
parent_path = os.getcwd()
directory = 'sequence/range_scale/normal_training/stride3_MAE_model2_turn_5_roll'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = output_dir + '/model.pth'

turn_threshold_file_per = output_dir + '/turn_threshold_per.pickle'
turn_mean_std_file_per = output_dir + '/turn_mean_std_per.pickle'
turn_threshold_file_abs = output_dir + '/turn_threshold_abs.pickle'
turn_mean_std_file_abs = output_dir + '/turn_mean_std_abs.pickle'

stable_threshold_file_per = output_dir + '/stable_threshold_per.pickle'
stable_mean_std_file_per = output_dir + '/stable_mean_std_per.pickle'
stable_threshold_file_abs = output_dir + '/stable_threshold_abs.pickle'
stable_mean_std_file_abs = output_dir + '/stable_mean_std_abs.pickle'

# load data
filename = 'log_17_2022-6-18-17-45-37_normal_data_last_degrees.csv'
read_path = '/home/mengjie/dataset/px4/fixed_wing/turn/5/'
df_train1 = pd.read_csv(read_path + filename)  # turn cut 3: 800 - 2450; 4: 1343:3736; 5: 1350:4100

# normalize data
X_training, nav_cmd_train = myModule.preprocess(df_train1)

# cut off the landing part
X_training = X_training[1350:4100]
X_training_normal = X_training.copy()
y_train = X_training_normal[:, y_index]
print(np.array(y_train).shape)
for i in range(1000):
    print(X_training[i][0])

# turning stage index
turn = [[1385, 1497], [1669, 1782], [1897, 1998], [2015, 2081], [2142, 2209], [2261, 2325], [2337, 2448], [2589, 2685], [2734, 2799],
        [2835, 2900], [2916, 3005], [3073, 3179], [3302, 3395], [3583, 3689], [3746, 3812], [3845, 3943], [3943, 4009], [4019, 4090]]

for pair in turn:
    print('#######')
    print(X_training[pair[0]-1350:pair[0]-1350+10, 0]*90)
# stable stage index
stable = [[1350, 1385], [1497, 1669], [1782, 1897], [1998, 2015], [2081, 2142], [2209, 2261], [2325, 2337], [2448, 2589],
          [2685, 2734], [2799, 2835], [2900, 2916], [3005, 3073], [3179, 3302], [3395, 3583], [3689, 3746], [3812, 3845], [4009, 4019], [4090, 4100]]

turn_x = []
turn_y = []

stable_x = []
stable_y = []

cut_index = 1350

for pair in turn:
    a = pair[0] - cut_index
    b = pair[1] - cut_index
    x = X_training[a:b]
    y = y_train[a:b]
    turn_x.append(x)
    turn_y.append(y)

for pair in stable:
    a = pair[0] - cut_index
    b = pair[1] - cut_index
    x = X_training[a:b]
    y = y_train[a:b]
    stable_x.append(x)
    stable_y.append(y)

df_x = pd.DataFrame(X_training)
df_y = pd.DataFrame(y_train)

df_x.to_csv(output_dir + '/df_x.csv')
df_y.to_csv(output_dir + '/df_y.csv')

index = range(30, len(X_training))
abnormal_indices = random.sample(index, 100)

# attack injection
X_training_abnormal = myModule.attack_test(X_training, 0, abnormal_indices)

print(np.array(X_training_abnormal).shape)
print(np.array(y_train).shape)


X_training_abnormal, y_training = myModule.training_reconstruct_data(X_training_abnormal, y_train, stride)

X_training_normal, y_training = myModule.training_reconstruct_data(X_training_normal, y_train, stride)


# X_training_normal = X_training
# y_training_normal = y_training
#
# X_training_abnormal = X_training_normal.copy()
# y_training_abnormal = y_training_normal.copy()

# split the training and validation dataset
X_training_normal, X_valid_normal, y_training_normal, y_valid_normal = train_test_split(X_training_normal,
                                                                                        y_training,
                                                                                        test_size=0.2, shuffle=False,
                                                                                        random_state=0)

X_training_abnormal, X_valid_abnormal, y_valid_abnormal, y_valid_abnormal = train_test_split(X_training_abnormal,
                                                                                                y_training,
                                                                                                test_size=0.2,
                                                                                                shuffle=False,
                                                                                                random_state=0)

# inject abnormal data
# if pattern != -1:
#     X_training_abnormal = myModule.abnormal_injection(X_training_abnormal, pattern, percentage, y_index)
#     X_valid_abnormal = myModule.abnormal_injection(X_valid_abnormal, pattern, percentage, y_index)

if pattern == 0:
    abnormal_pattern = 'continuously'
elif pattern == 1:
    abnormal_pattern = 'random'
elif pattern == -1:
    abnormal_pattern = 'normal'

# output log to txt file
sys.stdout = open(output_dir + '/output.txt', 'a')
print(f"######### training output {abnormal_pattern} ##########")

print("X_training.shape", X_training_abnormal.shape)
print("y_training.shape", y_training_normal.shape)

print("X_valid.shape", X_valid_abnormal.shape)
print("y_valid.shape", y_valid_normal.shape)

print("sliding window length", stride)

# log to terminal
sys.stdout.close()
sys.stdout = to_screen

# create lstm
mv_lstm = myModule.model2()
mv_lstm.float()
mv_lstm = mv_lstm.cuda()
optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)

# training model
min_valid_loss = np.inf
epoch_train_loss = []
MAE = torch.nn.L1Loss()

for t in range(train_episodes):
    mv_lstm.train()
    train_loss = []
    for i in range(0, len(X_training_normal), batch_size):
        inpt = X_training_normal[i:i + batch_size, :, :]
        target = y_training_normal[i:i + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)

        x_batch = x_batch.permute(1, 0, 2)
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        optimizer.zero_grad()
        output = mv_lstm(x_batch)

        # loss = myModule.custom_loss_func(output.view(-1), y_batch)
        loss = MAE(output.view(-1), y_batch)
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
turn_train_y_prediction = []
err_abs = torch.nn.L1Loss()
Filtered_err_percentage = []
low_pass_IIR = 0
percentage_low_pass_IIR = 0

# when computing threshold, feed the training data one by one, instead of by batch
turn_abs_err = []
turn_per_err = []

# turning threshold
for t in range(len(turn_x)):

    X_training_turn, y_training_turn = myModule.reconstruct_data(turn_x[t], y_index, stride)
    y_training_turn = y_training_turn * 90

    for i in range(0, len(X_training_turn)):
        inpt = [X_training_turn[i, :, :]]
        target = y_training_turn[i]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)
        x_batch = x_batch.permute(1, 0, 2)
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        output = my_lstm(x_batch)
        prediction = output.view(-1).to("cpu").detach().numpy()
        prediction = prediction * 90

        for k in range(len(prediction)):
            turn_train_y_prediction.append(prediction[k])

        err = abs(prediction - target)
        err_percentage = err.item() / abs(target)

        if i == 0:
            low_pass_IIR = myModule.LowPassIIR(myModule.b, err)
            percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage)

        filtered_err = low_pass_IIR.filter(err.item())
        Filtered.append(filtered_err.item())

        filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
        Filtered_err_percentage.append(filtered_err_percentage.item())

sys.stdout = open(output_dir + '/output.txt', 'a')

print("computing turning threshold")

filter_mean_abs = np.mean(Filtered)
filter_std_abs = np.std(Filtered)
FD_threshold_abs = filter_mean_abs + 2 * filter_std_abs

filter_mean_per = np.mean(Filtered_err_percentage)
filter_std_per = np.std(Filtered_err_percentage)
FD_threshold_per = filter_mean_per + 2 * filter_std_per

with open(turn_threshold_file_per, 'wb') as f:
    pickle.dump(FD_threshold_per, f)

with open(turn_mean_std_file_per, 'wb') as f:
    pickle.dump([filter_mean_per, filter_std_per], f)

with open(turn_threshold_file_abs, 'wb') as f:
    pickle.dump(FD_threshold_abs, f)

with open(turn_mean_std_file_abs, 'wb') as f:
    pickle.dump([filter_mean_abs, filter_std_abs], f)

print('turn threshold')
print("percentage error mean", filter_mean_per)
print("percentage error std", filter_std_per)
print("percentage FD_threshold", FD_threshold_per)

print("abs error mean", filter_mean_abs)
print("abs error std", filter_std_abs)
print("abs FD_threshold", FD_threshold_abs)

# compute stable threshold
stable_train_y_prediction = []

for t in range(len(stable_x)):

    X_training_stable, y_training_stable = myModule.reconstruct_data(stable_x[t], y_index, stride)
    y_training_turn = y_training_stable * 90

    for i in range(0, len(X_training_stable)):
        inpt = [X_training_stable[i, :, :]]
        target = y_training_stable[i]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)
        x_batch = x_batch.permute(1, 0, 2)
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        output = my_lstm(x_batch)
        prediction = output.view(-1).to("cpu").detach().numpy()
        prediction = prediction * 90

        for k in range(len(prediction)):
            stable_train_y_prediction.append(prediction[k])

        err = abs(prediction - target)
        err_percentage = err.item() / abs(target)

        if i == 0:
            low_pass_IIR = myModule.LowPassIIR(myModule.b, err)
            percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage)

        filtered_err = low_pass_IIR.filter(err.item())
        Filtered.append(filtered_err.item())

        filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
        Filtered_err_percentage.append(filtered_err_percentage.item())

print("computing stable threshold")

filter_mean_abs = np.mean(Filtered)
filter_std_abs = np.std(Filtered)
FD_threshold_abs = filter_mean_abs + 2 * filter_std_abs

filter_mean_per = np.mean(Filtered_err_percentage)
filter_std_per = np.std(Filtered_err_percentage)
FD_threshold_per = filter_mean_per + 2 * filter_std_per

with open(stable_threshold_file_per, 'wb') as f:
    pickle.dump(FD_threshold_per, f)

with open(stable_mean_std_file_per, 'wb') as f:
    pickle.dump([filter_mean_per, filter_std_per], f)

with open(stable_threshold_file_abs, 'wb') as f:
    pickle.dump(FD_threshold_abs, f)

with open(stable_mean_std_file_abs, 'wb') as f:
    pickle.dump([filter_mean_abs, filter_std_abs], f)

print('stable threshold')
print("percentage error mean", filter_mean_per)
print("percentage error std", filter_std_per)
print("percentage FD_threshold", FD_threshold_per)

print("abs error mean", filter_mean_abs)
print("abs error std", filter_std_abs)
print("abs FD_threshold", FD_threshold_abs)


Filtered = []
train_y_prediction = []
err_abs = torch.nn.L1Loss()
Filtered_err_percentage = []

y_training_normal = y_training_normal * 90
for i in range(0, len(X_training_normal)):
    inpt = [X_training_normal[i, :, :]]
    target = y_training_normal[i]

    x_batch = torch.tensor(inpt, dtype=torch.float32)
    y_batch = torch.tensor(target, dtype=torch.float32)
    x_batch = x_batch.permute(1, 0, 2)
    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()

    output = my_lstm(x_batch)
    prediction = output.view(-1).to("cpu").detach().numpy()
    prediction = prediction * 90

    for k in range(len(prediction)):
        train_y_prediction.append(prediction[k])

    err = abs(prediction - target)
    err_percentage = err.item() / abs(target)

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err)
        percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage)

    filtered_err = low_pass_IIR.filter(err.item())
    Filtered.append(filtered_err.item())

    filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
    Filtered_err_percentage.append(filtered_err_percentage.item())

print('Training MSE ', mean_squared_error(y_training_normal, train_y_prediction))

# compute validation accuracy
err_output = []
valid_err_percentage = []
valid_y_prediction = []
# y_valid_normal = y_scaler.inverse_transform(y_valid_normal)
y_valid_normal = y_valid_normal * 90
for i in range(0, len(X_valid_normal)):
    inpt = [X_valid_normal[i]]
    target = y_valid_normal[i]

    x_valid = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_valid = x_valid.cuda()
    y_target = y_target.cuda()

    x_valid = x_valid.permute(1, 0, 2)
    output = my_lstm(x_valid)
    prediction = output.view(-1).to("cpu").detach().numpy()
    prediction = prediction * 90

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

print('Validation MSE ', mean_squared_error(y_valid_normal, valid_y_prediction))

sys.stdout.close()
sys.stdout = to_screen

x_label = 'sliding window number'
y_label = 'roll angle'

# training result
myModule.two_line_1('training result', x_label, y_label, train_y_prediction, y_training_normal,
                    'training result',
                    output_dir)

# validation result
myModule.two_line_1('validation result', x_label, y_label, valid_y_prediction, y_valid_normal,
                    'validation result',
                    output_dir)

# test error
myModule.one_line('training error', x_label, y_label, Filtered, 'training error', output_dir)

print('finish')
