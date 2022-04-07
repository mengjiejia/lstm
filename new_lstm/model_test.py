import os
import pickle
import myModule
from myModule import Sequence
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import random

to_screen = sys.stdout

parent_path = os.getcwd()
directory = 'test_abnormal_training_decrease'
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
X_test = X_test[0:100]
y_test = y_test[0:100]
X_test_normal = X_test.copy()
y_test_normal = y_test.copy()
X_test_abnormal = X_test.copy()
y_test_abnormal = y_test.copy()

# attack pattern
pattern = 1
if pattern == 0:
    attack_pattern = 'continuously'
elif pattern == 1:
    attack_pattern = 'random'

sys.stdout = open(output_dir + '/output.txt', 'a')
print(f"######### test output {attack_pattern} ##########")
print("X_test.shape", X_test.shape)
print("y_test.shape", np.array(y_test).shape)


# inject abnormal data
# test_label = [0] * len(y_test_abnormal)
# count = 0

# for i in range(2000, 4000):
#     for j in range(stride):  # 3 is the roll_rate index
#     # for k in range(11):
#         X_test_abnormal[i][j][0] = X_test_abnormal[i][j][0] * random.uniform(1.1, 1.5)
#         y_test_abnormal[i] = y_test_abnormal[i] * random.uniform(1.1, 1.5)
#
#     test_label[i] = 1

X_test_abnormal, y_test_abnormal, abnormal_label, windows_count, abnormal_indices = myModule.attack(X_test_abnormal, y_test_abnormal, pattern, 0.2, y_index)
print("total abnormal windows", windows_count)

# number of data sources
n_features = 11

# this is length of sliding window.
n_steps = stride
batch_size = 16


# Load
my_lstm = torch.load(model_path)
my_lstm.eval()
err_abs = torch.nn.L1Loss()

# test
abnormalx_abnormaly_test_err = []
abnormalx_abnormaly_prediction = []

normalx_abnormaly_test_err = []
normalx_abnormaly_prediction = []

normalx_normaly_test_err = []
normalx_normaly_prediction = []
normal_label = [0] * len(y_test_normal)

# abnormal x, abnormal y
for i in range(0, len(X_test_abnormal)):
    inpt = [X_test_abnormal[i]]
    target = y_test_abnormal[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        abnormalx_abnormaly_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    abnormalx_abnormaly_test_err.append(test_filtered_loss)

# normal x, abnormal y
for i in range(0, len(X_test_normal)):
    inpt = [X_test_normal[i]]
    target = y_test_abnormal[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        normalx_abnormaly_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    normalx_abnormaly_test_err.append(test_filtered_loss)

# normal x, normal y
for i in range(0, len(X_test_normal)):
    inpt = [X_test_normal[i]]
    target = y_test_normal[i]

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    for k in range(len(prediction)):
        normalx_normaly_prediction.append(prediction[k])

    err = err_abs(output.view(-1), y_target)
    err = err.to("cpu").detach().numpy()
    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    normalx_normaly_test_err.append(test_filtered_loss)

# with open(threshold_file, 'rb') as f:
#     FD_threshold = pickle.load(f)

with open(mean_std_file, 'rb') as f:
    mean, std = pickle.load(f)

FD_threshold = (mean + 2 * std)
print('Fd_threshold', FD_threshold)

# abnormal x, abnormal y, accuracy
TP, FP, TN, FN, TPR, FPR, ACC, TP_list, FP_list, TN_list, FN_list = myModule.Accuracy(abnormal_label, abnormalx_abnormaly_test_err, FD_threshold)

test_positive = np.concatenate((TP_list, FP_list))
test_positive.sort()
abnormal_indices.sort()
TP_list.sort()
combined_positive_list = []
for item in abnormal_indices:
    if item in TP_list:
        combined_positive_list.append(1)
    else:
        combined_positive_list.append(0)

d = {'abnormal_indice': abnormal_indices, 'test': combined_positive_list}
df_positive_list = pd.DataFrame(d)

print('positive_list')
print(df_positive_list)


print('abnormal x, abnormal y, test results:')
print("TP", TP)
print("FP", FP)
print("TN", TN)
print("FN", FN)
print("TPR", TPR)
print("FPR", FPR)
print("ACC", ACC)

# # normal x, abnormal y, accuracy
# TP, FP, TN, FN, TPR, FPR, ACC = myModule.Accuracy(abnormal_label, normalx_abnormaly_test_err, FD_threshold)
# print('normal x, abnormal y, test results:')
# print("TP", TP)
# print("FP", FP)
# print("TN", TN)
# print("FN", FN)
# print("TPR", TPR)
# print("FPR", FPR)
# print("ACC", ACC)
#
# # normal accuracy
# TP, FP, TN, FN, TPR, FPR, ACC = myModule.Accuracy(normal_label, normalx_normaly_test_err, FD_threshold)
# print('normal x, normal y, test results:')
# print("TP", TP)
# print("FP", FP)
# print("TN", TN)
# print("FN", FN)
# print("TPR", TPR)
# print("FPR", FPR)
# print("ACC", ACC)


x_label = 'sliding window number'
y_label = 'roll angle'

abnormal_mark = []
for item in abnormal_indices:
    if item < 100:
        abnormal_mark.append(item)

test_mark = []
for item in test_positive:
    if item < 100:
        test_mark.append(item)

# test result
# myModule.three_line('test result', x_label, y_label, test_y_prediction, y_test, y_test_abnormal, 'test result', output_dir)
# abnormal x, abnormal y, results
myModule.test_two_line('test result', x_label, y_label, abnormalx_abnormaly_prediction[0:100], y_test_abnormal[0:100], 0, 0, 'test result',
                       output_dir, abnormal_mark, TP_list, FP_list)
myModule.two_line('random abnormal_abnormal test result with normal sensor reading', x_label, y_label, abnormalx_abnormaly_prediction[0:100], y_test_normal[0:100],
                  'random abnormal_abnormal test result with normal sensor reading', output_dir)

# normal x, abnormal y, results
# myModule.test_two_line('normal x, abnormal y, test result', x_label, y_label, normalx_abnormaly_prediction[0:100], y_test_abnormal, 0, 0, 'normal x, abnormal y, test result',
#                        output_dir)
myModule.two_line('normal_abnormal test result with normal sensor reading', x_label, y_label, normalx_abnormaly_prediction, y_test_normal,
                  'normal_abnormal test result with normal sensor reading', output_dir)

# normal x, normal y, test results
# myModule.test_two_line('normal x, normal y, test result random', x_label, y_label, normalx_normaly_prediction, y_test_normal, 0, 0, 'normal test result',
#                        output_dir)


print('MSE abnormal x, abnormal y', mean_squared_error(y_test_normal, abnormalx_abnormaly_prediction))
print('MSE normal x, abnormal y', mean_squared_error(y_test_normal, normalx_abnormaly_prediction))
print('MSE normal x, normal y', mean_squared_error(y_test_normal, normalx_normaly_prediction))

sys.stdout.close()
sys.stdout = to_screen
print('finish')
