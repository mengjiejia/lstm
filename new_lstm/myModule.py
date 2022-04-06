import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from matplotlib import pyplot as plt


# lstm model
class Sequence(torch.nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = torch.nn.LSTMCell(11, 128)
        self.drop_out1 = torch.nn.Dropout(0.2)
        self.lstm2 = torch.nn.LSTMCell(128, 64)
        self.drop_out2 = torch.nn.Dropout(0.2)
        self.lstm3 = torch.nn.LSTMCell(64, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 1)

        self.device = torch.device('cuda')

    def custom_loss_func(y_predictions, target):
        square_difference = torch.square(y_predictions - target)
        loss_value = torch.sum(square_difference) * 0.5
        return loss_value

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

        return output


# training and validation dataset abnormal injection
# stride = 5
def abnormal_injection(data, pattern, percentage, index, rate):
    random.seed(5)
    p = int(len(data) * percentage)
    abnormal_indices = random.sample(range(len(data)), k=p)

    # continuously
    if pattern == 0:
        for i in range(300, 1581):
            for j in range(5):
                data[i][j][index] = data[i][j][index] * rate

    # randomly
    elif pattern == 1:
        for i in abnormal_indices:
            rate = random.uniform(1.1, 1.5)
            for j in range(5):
                data[i][j][index] = data[i][j][index] * rate

    return data


# test dataset attack, label
def attack(X_test, y_test, pattern, percentage, index):
    random.seed(5)
    p = int(len(y_test) * percentage)
    abnormal_indices = random.sample(range(len(y_test)), k=p)
    test_label = [0] * len(y_test)
    # continuously
    if pattern == 0:
        for i in range(2000, 3400):
            rate = random.uniform(1.1, 1.5)
            for j in range(5):  # 3 is the roll_rate index
                X_test[i][j][index] = X_test[i][j][index] * rate

            y_test[i] = y_test[i] * rate
            test_label[i] = 1

    # random
    elif pattern == 1:
        for i in abnormal_indices:
            rate = random.uniform(1.1, 1.5)
            for j in range(5):
                X_test[i][j][index] = X_test[i][j][index] * rate

            y_test[i] = y_test[i] * rate
            test_label[i] = 1

    return X_test, y_test, test_label, p


# multivariate data preparation

def preprocess(df, y_index):
    df.rename(columns={'Unnamed: 0': 'datasource'}, inplace=True)
    df = df.drop(columns=['datasource'])
    X = df.to_numpy()
    # X = np.delete(X, y_index, 0)  # remove the predicted sensor
    y = X[y_index]
    X = X.T

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
    if TP == 0 and FN == 0:
        TPR = 0
    else:
        TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + TN + FP + FN)

    return TP, FP, TN, FN, TPR, FPR, ACC


def test_two_line(title, xlabel, ylabel, prediction, abnormal, start, end, filename, path):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(len(prediction)), prediction, 'r', linewidth=2.0, label='prediction')
    plt.plot(np.arange(len(abnormal)), abnormal, 'g', linewidth=2.0, label='abnormal')
    # add vertical line
    x_position = [start, end]
    for i in x_position:
        plt.axvline(x=i, color='y', linestyle='--')
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
