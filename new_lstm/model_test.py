import os
import pickle
import myModule
from myModule import Sequence
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import sys

to_screen = sys.stdout

parent_path = os.getcwd()
directory = '20_percentage_05'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = output_dir + '/model.pth'
threshold_file = output_dir + '/threshold.pickle'
mean_std_file = output_dir + '/mean_std.pickle'

# load test data
df_test = pd.read_csv('log_8_2022-3-23-11-40-47_normal_data_last.csv')
y_index = 0
X_test, y_test = myModule.preprocess(df_test, y_index)
X_test = myModule.normalize(X_test)
y_test = myModule.normalize(y_test)

# sliding window length
stride = 5

X_test, y_test = myModule.reconstruct_data(X_test, y_test, stride)

# cut the landing part
X_test = X_test[0:10000]
y_test = y_test[0:10000]
X_test_normal = X_test.copy()

sys.stdout = open(output_dir + '/output.txt', 'a')
print("######### test output ##########")
print("X_test.shape", X_test.shape)
print("y_test.shape", np.array(y_test).shape)
sys.stdout.close()
sys.stdout = to_screen

# inject abnormal data
y_test_abnormal = y_test.copy()
test_label = [0] * len(y_test_abnormal)
for i in range(2000, 4000):
    for j in range(stride):  # 3 is the roll_rate index
        # for k in range(11):
        X_test[i][j][0] = X_test[i][j][0] * 0.5
    y_test_abnormal[i] = y_test_abnormal[i] * 0.5

    test_label[i] = 1

# number of data sources
n_features = 11

# this is length of sliding window.
n_steps = stride
batch_size = 16

sys.stdout = open(output_dir + '/output.txt', 'a')

# Load
my_lstm = torch.load(model_path)
my_lstm.eval()
err_abs = torch.nn.L1Loss()

# test
test_err = []
test_y_prediction = []

normal_test_err = []
normal_test_y_prediction = []

for i in range(0, len(X_test)):
    inpt = [X_test[i]]
    target = y_test_abnormal[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        test_y_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    test_err.append(test_filtered_loss)

for i in range(0, len(X_test)):
    inpt = [X_test_normal[i]]
    target = y_test[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        normal_test_y_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    normal_test_err.append(test_filtered_loss)

# with open(threshold_file, 'rb') as f:
#     FD_threshold = pickle.load(f)

with open(mean_std_file, 'rb') as f:
    mean, std = pickle.load(f)

FD_threshold = (mean + 2 * std) * 0.75

TP, FP, TN, FN, TPR, FPR, ACC = myModule.Accuracy(test_label, test_err, FD_threshold)
print("TP", TP)
print("FP", FP)
print("TN", TN)
print("FN", FN)
print("TPR", TPR)
print("FPR", FPR)
print("ACC", ACC)

# print('normal test data regression accuracy')
# normal_test_label = [0] * len(y_test)
# TP, FP, TN, FN, TPR, FPR, ACC = myModule.Accuracy(normal_test_label, normal_test_err, FD_threshold)
# print("TP", TP)
# print("FP", FP)
# print("TN", TN)
# print("FN", FN)
# print("TPR", TPR)
# print("FPR", FPR)
# print("ACC", ACC)

x_label = 'sliding window number'
y_label = 'roll angle'

# test result
# myModule.three_line('test result', x_label, y_label, test_y_prediction, y_test, y_test_abnormal, 'test result', output_dir)
myModule.test_two_line('test result', x_label, y_label, test_y_prediction, y_test_abnormal, 2000, 4000, 'test result',
                       output_dir)
myModule.two_line('test result with normal sensor reading', x_label, y_label, test_y_prediction, y_test,
                  'test result with normal sensor reading', output_dir)

sys.stdout.close()
sys.stdout = to_screen
print('finish')
