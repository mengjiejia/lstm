import os
import pickle
import myModule
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import sys
from joblib import dump, load
import random

to_screen = sys.stdout

parent_path = os.getcwd()
directory = 'sequence/range_scale/normal_training/MAE_model2_turn_5_roll'
output_dir = os.path.join(parent_path, directory)

# Check whether the specified path exists or not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = output_dir + '/model.pth'
turn_model_path = output_dir + '/turn_model.pth'

turn_threshold_file_abs = output_dir + '/turn_threshold_abs.pickle'
turn_mean_std_file_abs = output_dir + '/turn_mean_std_abs.pickle'

stable_threshold_file_abs = output_dir + '/stable_threshold_abs.pickle'
stable_mean_std_file_abs = output_dir + '/stable_mean_std_abs.pickle'

begin_turn_mean_std_file = output_dir + '/begin_turn_model_mean_std.pickle'

# sliding window length
stride = 5

# predicted sensor
y_index = 0

# attack pattern
pattern = 1

# number of data sources, remove altitude
n_features = 10

# model parameters
n_steps = stride
batch_size = 16

# load test data
df_test = pd.read_csv(
    '/home/mengjie/dataset/px4/fixed_wing/turn/4/log_7_2022-6-14-18-53-47_normal_data_last_degrees.csv')  # cut 1330:3736

X_test, nav_test = myModule.preprocess(df_test)

# cut the landing part
X_test = X_test[1330:3736]
nav_test = nav_test[1330:3736]

r = range(50, 2400)
abnormal_indices = random.sample(r, 100)

X_test_normal = X_test.copy()
y_test_normal = np.array(X_test_normal[:, y_index]) * 90

X_test_abnormal = myModule.attack_test(X_test, 0, abnormal_indices)

if pattern == 0:
    attack_pattern = 'continuously'
elif pattern == 1:
    attack_pattern = 'randomly increase'
elif pattern == 2:
    attack_pattern = 'randomly decrease'
elif pattern == 3:
    attack_pattern = 'randomly multiply'

sys.stdout = open(output_dir + '/output.txt', 'a')
print(f"######### test output {attack_pattern} ##########")
print("X_test.shape", X_test.shape)
print("y_test.shape", np.array(y_test_normal).shape)

print('abnormal datapoints')
print(sorted(abnormal_indices))

windows_count = len(abnormal_indices)
print("total abnormal datapoints", windows_count)

# Load
my_lstm = torch.load(model_path)
my_lstm.eval()
err_abs = torch.nn.L1Loss()

Filtered = []
train_y_prediction = []

Filtered_err_percentage = []
low_pass_IIR = 0
percentage_low_pass_IIR = 0

turning = False

turning_point_count = 0

# load turn regression model
turn_lstm = torch.load(turn_model_path)

# test
abnormalx_abnormaly_test_err = []
abnormalx_abnormaly_raw_err = []
abnormalx_abnormaly_prediction = []
abnormalx_abnormaly_test_err_per = []
abnormalx_abnormaly_positive = []

normalx_abnormaly_test_err = []
normalx_abnormaly_prediction = []
normalx_abnormaly_test_err_per = []

normalx_normaly_test_err = []
normalx_normaly_prediction = []
normalx_normaly_test_err_per = []

with open(turn_threshold_file_abs, 'rb') as f:
    turn_FD_threshold_abs = pickle.load(f)

with open(turn_mean_std_file_abs, 'rb') as f:
    turn_mean_abs, turn_std_abs = pickle.load(f)

with open(stable_threshold_file_abs, 'rb') as f:
    stable_FD_threshold_abs = pickle.load(f)

with open(stable_mean_std_file_abs, 'rb') as f:
    stable_mean_abs, stable_std_abs = pickle.load(f)

with open(begin_turn_mean_std_file, 'rb') as f:
    begin_turn_mean_abs, begin_turn_std_abs = pickle.load(f)

turn_FD_threshold_abs = (turn_mean_abs + 2 * turn_std_abs)
stable_FD_threshold_abs = (stable_mean_abs + 2 * stable_std_abs)

begin_turn_FD_threshold = (begin_turn_mean_abs + 1 * begin_turn_std_abs)
FD_threshold = stable_FD_threshold_abs

print('turn_FD_threshold_abs', turn_FD_threshold_abs)
print('stable_FD_threshold_abs', stable_FD_threshold_abs)

y_test_abnormal = X_test_abnormal[:, y_index] * 90

for i in range(0, len(X_test_abnormal) - stride):

    inpt = [X_test_abnormal[i:i + stride, :]]
    inpt = np.array(inpt)
    target = X_test_abnormal[i + stride][y_index] * 90

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    # waypoint if nav_test = 16
    if nav_test[i + stride] == 16:
        turning_point_count = 0
        turning = True
        print('turning:', i + stride)
        print('begin_turn_FD_threshold', begin_turn_FD_threshold)

    elif abs(X_test_abnormal[i + stride][5] * 180) <= 0.5 and 16 not in nav_test[i:i + stride]:
        turning = False
        # set turning point count = 0
        turning_point_count = 0
        FD_threshold = stable_FD_threshold_abs
        print('stable:', i + stride)

    if turning:

        # if the first prediction is the start point of a turn, then initialize the low_pass_IIR with error = 0
        if i == 0:
            low_pass_IIR = myModule.LowPassIIR(myModule.b, 0)

        turning_point_count += 1

        if turning_point_count <= 5:
            if turning_point_count <= 3:
                print(f'{turning_point_count} point turn, using sensor reading as prediction', i + stride)
                abnormalx_abnormaly_prediction.append(X_test_abnormal[i + stride][y_index] * 90)
                err = 0
                abnormalx_abnormaly_raw_err.append(err)
                filtered_err_abs = low_pass_IIR.filter(err)
                abnormalx_abnormaly_test_err.append(filtered_err_abs.item())

            else:
                print(f'{turning_point_count} point turn, using regression model to predict value', i + stride)

                inpt = [X_test_abnormal[i + stride - 2:i + stride, :]]
                inpt = np.array(inpt)

                x_test = torch.tensor(inpt, dtype=torch.float32)
                x_test = x_test.permute(1, 0, 2)
                x_test = x_test.cuda()

                output = turn_lstm(x_test)
                prediction = output.view(-1).to("cpu").detach().numpy()
                prediction = prediction.reshape((len(prediction), 1))
                prediction = prediction * 90
                prediction = prediction.reshape((len(prediction),))

                err = abs(prediction - target)
                abnormalx_abnormaly_raw_err.append(err)

                filtered_err_abs = low_pass_IIR.filter(err)
                abnormalx_abnormaly_test_err.append(filtered_err_abs.item())

                if filtered_err_abs > begin_turn_FD_threshold:
                    print('abnormal turning', i + stride)
                    print('filtered_err_abs', filtered_err_abs)
                    print('prediction', prediction)
                    print('sensor reading', target)
                    # replace abnormal datapoint by predicted value
                    X_test_abnormal[i + stride][y_index] = prediction / 90
                    abnormalx_abnormaly_positive.append(i + stride)

                abnormalx_abnormaly_prediction.append(prediction)

        # switch back to the original lstm model after the first 5 data points
        else:

            FD_threshold = turn_FD_threshold_abs

            output = my_lstm(x_test)
            prediction = output.view(-1).to("cpu").detach().numpy()
            prediction = prediction.reshape((len(prediction), 1))
            prediction = prediction * 90
            prediction = prediction.reshape((len(prediction),))

            err = abs(prediction - target)
            abnormalx_abnormaly_raw_err.append(err.item())

            filtered_err_abs = low_pass_IIR.filter(err.item())
            abnormalx_abnormaly_test_err.append(filtered_err_abs.item())

            if filtered_err_abs > FD_threshold:
                print('abnormal turning', i + stride)
                print('filtered_err_abs', filtered_err_abs)
                print('prediction', prediction)
                print('sensor reading', target)
                # replace abnormal datapoint by predicted value
                X_test_abnormal[i + stride][y_index] = prediction / 90
                abnormalx_abnormaly_positive.append(i + stride)

            abnormalx_abnormaly_prediction.append(prediction[0])

    # stable
    else:

        FD_threshold = stable_FD_threshold_abs

        output = my_lstm(x_test)
        prediction = output.view(-1).to("cpu").detach().numpy()
        prediction = prediction.reshape((len(prediction), 1))
        prediction = prediction * 90
        prediction = prediction.reshape((len(prediction),))

        err = abs(prediction - target)
        abnormalx_abnormaly_raw_err.append(err.item())

        if i == 0:
            low_pass_IIR = myModule.LowPassIIR(myModule.b, err)

        filtered_err_abs = low_pass_IIR.filter(err.item())
        abnormalx_abnormaly_test_err.append(filtered_err_abs.item())

        if filtered_err_abs > FD_threshold:
            print('abnormal stable', i + stride)
            print('filtered_err_abs', filtered_err_abs)
            # replace abnormal datapoint by predicted value
            print('prediction', prediction)
            print('sensor reading', target)

            X_test_abnormal[i + stride][y_index] = prediction / 90
            abnormalx_abnormaly_positive.append(i + stride)

        abnormalx_abnormaly_prediction.append(prediction[0])

# normal x, normal y
for i in range(0, len(X_test_normal) - stride):

    # waypoint if nav_test = 16
    if nav_test[i + stride] == 16:
        turning_point_count = 0
        turning = True
        FD_threshold = begin_turn_FD_threshold
        print('turning:', i + stride)
        print('begin_turn_FD_threshold', begin_turn_FD_threshold)
    # X_test_abnormal[i + stride][y_index] * 90
    elif abs(X_test_abnormal[i + stride][5] * 180) <= 0.5 and 16 not in nav_test[i:i + stride]:
        turning = False
        # set turning point count = 0
        turning_point_count = 0
        FD_threshold = stable_FD_threshold_abs
        print('stable:', i + stride)

    inpt = [X_test_normal[i:i + stride, :]]
    target = X_test_normal[i + stride][y_index] * 90
    inpt = np.array(inpt)

    x_test = torch.tensor(inpt, dtype=torch.float32)
    y_target = torch.tensor(target, dtype=torch.float32)
    x_test = x_test.permute(1, 0, 2)
    x_test = x_test.cuda()
    y_target = y_target.cuda()

    output = my_lstm(x_test)
    prediction = output.view(-1).to("cpu").detach().numpy()
    prediction = prediction.reshape((len(prediction), 1))
    prediction = prediction * 90
    prediction = prediction.reshape((len(prediction),))

    for k in range(len(prediction)):
        normalx_normaly_prediction.append(prediction[k])

    err = abs(prediction - target)
    err_percentage = err.item() / abs(prediction)

    if i == 0:
        low_pass_IIR = myModule.LowPassIIR(myModule.b, err.item())
        percentage_low_pass_IIR = myModule.LowPassIIR(myModule.b, err_percentage.item())

    test_filtered_loss = low_pass_IIR.filter(err.item())
    normalx_normaly_test_err.append(test_filtered_loss)

    filtered_err_percentage = percentage_low_pass_IIR.filter(err_percentage.item())
    normalx_normaly_test_err_per.append(filtered_err_percentage)

# abnormal x, abnormal y, accuracy
TP, FP, TN, FN, TPR, FPR, ACC, TP_list, FP_list, TN_list, FN_list = myModule.Accuracy(abnormal_indices,
                                                                                      abnormalx_abnormaly_positive,
                                                                                      len(X_test_abnormal))

print('FP', sorted(FP_list))
print('FN', sorted(FN_list))

print('abnormal x, abnormal y, test results:')
print("TP", TP)
print("FP", FP)
print("TN", TN)
print("FN", FN)
print("TPR", TPR)
print("FPR", FPR)
print("ACC", ACC)

x_label = 'data points'
y_label = 'roll angle'

myModule.mark_two_line(f'{attack_pattern} test result with abnormal reading', x_label, y_label,
                       abnormalx_abnormaly_prediction,
                       y_test_abnormal, 0, 0, f'{attack_pattern} test result with abnormal reading',
                       output_dir, abnormal_indices, TP_list, FP_list, stride)
myModule.two_line(f'{attack_pattern} test result with normal reading', x_label, y_label, abnormalx_abnormaly_prediction,
                  y_test_normal,
                  f'{attack_pattern} test result with normal reading', output_dir, stride, FP_list)

# normal x, abnormal y, results

myModule.two_line('normal test result', x_label, y_label, normalx_normaly_prediction, y_test_normal,
                  'normal test result', output_dir, stride, [])

print('MSE abnormal x, abnormal y', mean_squared_error(y_test_normal[stride:], abnormalx_abnormaly_prediction))
print('MSE normal x, normal y', mean_squared_error(y_test_normal[stride:], normalx_normaly_prediction))

abnormalx_abnormaly_prediction = [None] * stride + abnormalx_abnormaly_prediction
normalx_normaly_prediction = [None] * stride + normalx_normaly_prediction
abnormalx_abnormaly_test_err = [None] * stride + abnormalx_abnormaly_test_err
abnormalx_abnormaly_raw_err = [None] * stride + abnormalx_abnormaly_raw_err

df = pd.DataFrame(data={"sensor reading": y_test_abnormal, "prediction": abnormalx_abnormaly_prediction})
df.to_csv(output_dir + '/attack.csv', sep=',')

df = pd.DataFrame(data={"sensor reading": y_test_normal, "prediction": normalx_normaly_prediction})
df.to_csv(output_dir + '/normal.csv', sep=',')

print('len(abnormalx_abnormaly_raw_err)', len(abnormalx_abnormaly_raw_err))
print('len(abnormalx_abnormaly_test_err)', len(abnormalx_abnormaly_test_err))

df = pd.DataFrame(data={"raw error": abnormalx_abnormaly_raw_err, "filtered error": abnormalx_abnormaly_test_err})
df.to_csv(output_dir + '/error.csv', sep=',')

sys.stdout.close()
sys.stdout = to_screen
print('finish')
