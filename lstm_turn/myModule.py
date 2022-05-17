import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


# lstm model

def custom_loss_func(y_predictions, target):
    square_difference = torch.square(y_predictions - target)
    loss_value = torch.sum(square_difference) * 0.5
    return loss_value


class Sequence(torch.nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = torch.nn.LSTMCell(10, 128)
        self.drop_out1 = torch.nn.Dropout(0.2)
        self.lstm2 = torch.nn.LSTMCell(128, 64)
        self.drop_out2 = torch.nn.Dropout(0.2)
        self.lstm3 = torch.nn.LSTMCell(64, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 1)

        self.device = torch.device('cuda')

    def forward(self, input):
        # Initial cell states, every time training batch?
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
            h_t1, c_t1 = self.lstm1(input[i], (h_t1, c_t1))
            h_t1 = self.drop_out1(h_t1)
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t2 = self.drop_out2(h_t2)
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.fc1(h_t3)
            output = self.fc2(output)

        # last sample as input
        h_t1, c_t1 = self.lstm1(input[4], (h_t1, c_t1))

        h_t1 = self.drop_out1(h_t1)
        h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
        h_t2 = self.drop_out2(h_t2)
        h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
        output = self.fc1(h_t3)
        output = self.fc2(output)

        return output


# model2
class model2(torch.nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.lstm1 = torch.nn.LSTMCell(10, 8)
        self.lstm2 = torch.nn.LSTMCell(8, 8)
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, 1)

        self.device = torch.device('cuda')

    def forward(self, input):
        # Initial cell states, every time training batch?
        h_t1 = torch.zeros(input.size(1), 8, dtype=torch.float32).to(self.device)
        c_t1 = torch.zeros(input.size(1), 8, dtype=torch.float32).to(self.device)
        h_t2 = torch.zeros(input.size(1), 8, dtype=torch.float32).to(self.device)
        c_t2 = torch.zeros(input.size(1), 8, dtype=torch.float32).to(self.device)

        outputs = []
        batch_size = input.size(1)
        seq_length = input.size(0)
        input = input.view(seq_length, batch_size, -1)
        for i in range(input.size(0)):
            h_t1, c_t1 = self.lstm1(input[i], (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.fc1(h_t2)
            output = self.fc2(output)

        # last sample as input
        h_t1, c_t1 = self.lstm1(input[4], (h_t1, c_t1))
        h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
        output = self.fc1(h_t2)
        output = self.fc2(output)

        return output


# training and validation dataset abnormal injection
# stride = 5
def abnormal_injection(data, pattern, percentage, index):
    random.seed(5)
    p = int(len(data) * percentage)
    abnormal_indices = random.sample(range(len(data)), k=p)

    # continuously
    if pattern == 0:
        for i in range(300, 1581):
            # value in degree
            value = random.uniform(0.5, 1.0)
            for j in range(5):
                data[i][j][index] = data[i][j][index] + value / 90  # value/90: scale the injection value

    # randomly increase
    elif pattern == 1:
        for i in abnormal_indices:
            value = random.uniform(1.0, 5.0)
            for j in range(5):
                data[i][j][index] = data[i][j][index] + value / 90

    # randomly decrease
    elif pattern == 2:
        for i in abnormal_indices:
            value = random.uniform(0.5, 1.0)
            for j in range(5):
                data[i][j][index] = data[i][j][index] - value / 90

    # randomly multiply
    elif pattern == 3:
        for i in abnormal_indices:
            rate = random.uniform(1.5, 2.0)
            for j in range(5):
                data[i][j][index] = data[i][j][index] * rate

    return data


# test dataset attack, label
def attack(X_test, y_test, pattern, percentage, index):
    random.seed(5)
    p = int(len(y_test) * percentage)
    test_label = [0] * len(y_test)

    # continuously
    if pattern == 0:
        abnormal_indices = np.arange(75, 150, 1)
        for i in range(75, 150):
            value = random.uniform(0.05, 0.1)
            for j in range(5):
                X_test[i][j][index] = X_test[i][j][index] + value / 90
            y_test[i] = y_test[i] + value / 90
            test_label[i] = 1

    # random increase
    elif pattern == 1:
        abnormal_indices = random.sample(range(len(y_test)), k=p)
        for i in abnormal_indices:
            value = random.uniform(1.0, 5.0)
            for j in range(5):
                X_test[i][j][index] = X_test[i][j][index] + value / 90
            y_test[i] = y_test[i] + value / 90
            test_label[i] = 1

    # random decrease
    elif pattern == 2:
        abnormal_indices = random.sample(range(len(y_test)), k=p)
        for i in abnormal_indices:
            value = random.uniform(5.0, 10.0)
            for j in range(5):
                X_test[i][j][index] = X_test[i][j][index] - value / 90

            y_test[i] = y_test[i] - value / 90
            test_label[i] = 1

    # random multiply
    elif pattern == 3:
        abnormal_indices = random.sample(range(len(y_test)), k=p)
        for i in abnormal_indices:
            rate = random.uniform(0.5, 1.0)
            for j in range(5):
                X_test[i][j][index] = X_test[i][j][index] * rate

            y_test[i] = y_test[i] * rate
            test_label[i] = 1

    return X_test, y_test, test_label, p, abnormal_indices


def preprocess(df, y_index):
    df.rename(columns={'Unnamed: 0': 'datasource'}, inplace=True)
    df = df.drop(columns=['datasource'])
    X = df.to_numpy()
    X = X.T
    X = np.delete(X, 10, 1)
    # roll, roll_rate, pitch, pitch_rate, act_roll, atc_pitch, divided by 90 degree
    for i in [0, 1, 3, 4, 6, 7]:
        X[:, i] = X[:, i] / 90

    # yaw, yaw_rate, act_yaw, divided by 180 degree
    for i in [2, 5, 8]:
        X[:, i] = X[:, i] / 180

    # airspeed divided by 30
    X[:, 9] = X[:, 9] / 30

    y = X[:, y_index]
    return X, y


def reconstruct_data(X_train, y_train, stride):
    X = []
    y = []
    L = len(X_train)
    for i in range(L - stride):
        train_x = X_train[i:i + stride, :]
        train_y = y_train[i + stride:i + stride + 1]
        X.append(train_x)
        y.append(train_y)
    return np.array(X), np.array(y)


# use 5 to predict 3
def reconstruct_data_middle(X_train, y_train, stride):
    X = []
    y = []
    L = len(X_train)
    for i in range(L - stride + 1):
        train_x = X_train[i:i + stride, :]
        train_y = y_train[i + 2:i + 2 + 1]
        X.append(train_x)
        y.append(train_y)
    return np.array(X), np.array(y)


# use 5 to predict 4
def reconstruct_data_4(X_train, y_train, stride):
    X = []
    y = []
    L = len(X_train)
    for i in range(L - stride + 1):
        train_x = X_train[i:i + stride, :]
        train_y = y_train[i + 3:i + 3 + 1]
        X.append(train_x)
        y.append(train_y)
    return np.array(X), np.array(y)


# use 5 to predict 5
def reconstruct_data_last(X_train, y_train, stride):
    X = []
    y = []
    L = len(X_train)
    for i in range(L - stride + 1):
        train_x = X_train[i:i + stride, :]
        train_y = y_train[i + stride - 1:i + stride]
        X.append(train_x)
        y.append(train_y)
    return np.array(X), np.array(y)


# normalize data using zscore
def normalize(data):
    data = zscore(data, axis=0)
    data = np.nan_to_num(data)
    return data


# random shuffle
def random_shuffle(X, y):
    xy_merged = []
    for i in range(len(X)):
        item = [X[i], y[i]]
        item = np.array(item)
        xy_merged.append(item)

    xy_merged = np.array(xy_merged)

    random.shuffle(xy_merged)

    X_random = []
    y_random = []
    for item in xy_merged:
        X_random.append(item[0])
        y_random.append(item[1])

    X_random = np.array(X_random)
    y_random = np.array(y_random)

    return X_random, y_random


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
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    for i in range(len(filter_loss)):
        # abnormal, positive
        if filter_loss[i] >= threshold:
            if label[i] == 1:
                TP += 1
                TP_list.append(i)
            else:
                FP += 1
                FP_list.append(i)
        else:
            # normal, negative
            if label[i] == 0:
                TN += 1
                TN_list.append(i)
            else:
                FN += 1
                FN_list.append(i)
    if TP == 0 and FN == 0:
        TPR = 0
    else:
        TPR = TP / (TP + FN)
        # print(TP)
        # print(FN)
    if FP == 0 and TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)

    ACC = (TP + TN) / (TP + TN + FP + FN)

    return TP, FP, TN, FN, TPR, FPR, ACC, TP_list, FP_list, TN_list, FN_list


def mark_two_line(title, xlabel, ylabel, prediction, abnormal, start, end, filename, path, abnormal_mark, TP_list,
                  FP_list):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(len(prediction)), prediction, 'r', marker='o', markevery=TP_list, linewidth=2.0,
             label='TP_prediction')
    plt.plot(np.arange(len(prediction)), prediction, 'r', marker='s', markevery=FP_list, linewidth=2.0,
             label='FP_prediction')
    plt.plot(np.arange(len(abnormal)), abnormal, 'g', marker='D', markevery=abnormal_mark, linewidth=2.0,
             label='abnormal')
    # add vertical line
    x_position = [start, end]
    for i in x_position:
        plt.axvline(x=i, color='y', linestyle='--')

    pair = []
    # for i in abnormal_mark:
    #     pair.append((prediction[i], abnormal[i]))

    for i in TP_list:
        pair.append((prediction[i], abnormal[i]))

    # plt.plot((abnormal_mark, abnormal_mark), ([i for (i, j) in pair], [j for (i, j) in pair]), color='y',
    #          linestyle='--')

    plt.plot((TP_list, TP_list), ([i for (i, j) in pair], [j for (i, j) in pair]), color='y',
             linestyle='--')

    lgd = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.legend()
    plt.savefig(path + '/' + filename + '.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def two_line(title, xlabel, ylabel, prediction, normal, filename, path):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(len(prediction)), prediction, 'r', linewidth=2.0, label='prediction')
    plt.plot(np.arange(len(normal)), normal, 'b', linewidth=2.0, label='normal_sensor_reading')
    lgd = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.legend()
    plt.savefig(path + '/' + filename + '.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def one_line(title, xlabel, ylabel, data, filename, path):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(len(data)), data, 'r', linewidth=2.0, label='error')
    lgd = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.legend()
    plt.savefig(path + '/' + filename + '.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def windows(title, xlabel, ylabel, prediction, abnormal, index_list, filename, path):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(index_list, prediction, 'r', linewidth=2.0, label='prediction')
    plt.plot(index_list, abnormal, 'b', linewidth=2.0, label='abnormal_sensor_reading')
    lgd = plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.legend()
    plt.savefig(path + '/' + filename + '.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
