import numpy as np
import matplotlib.pyplot as plt
import random

TRAINING_FILE = 'horseColicTraining.txt'
TESTING_FILE = 'horseColicTest.txt'
NUM_ITERATORS = 1000
INIT_STEP = 0.01

def load_data_set(filename):
    data_mat = [];label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        lin_arr = line.strip().split()
        data_mat.append([1, float(lin_arr[0]), float(lin_arr[1])])
        label_mat.append(int(lin_arr[2]))
    return data_mat, label_mat

def sigmoid(in_x):
    return 1.0/(1 + np.exp(-in_x))

def grad_ascent(data, targets, step, iter_times):
    data_mat = np.mat(data)
    targets = np.mat(targets).transpose()
    m,n = data_mat.shape
    weights = np.ones((n,1))
    for i in range(iter_times):
        h = sigmoid(data_mat*weights)
        error = targets - h
        weights = weights + step * data_mat.transpose() * error
    return weights

def stoc_grad_ascent(data, labels, num_iter):
    m, n = np.shape(data)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            step = 4/(1.0 + i + j) + INIT_STEP
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data[rand_index]*weights))
            error = labels[rand_index] - h
            weights = weights + step * error * data[rand_index]
            del(data_index[rand_index])
    return weights

def colic_test():
    fr_train = open(TRAINING_FILE)
    fr_test = open(TESTING_FILE)
    train_set = [];train_labels = []
    for line in fr_train.readlines():
        cur_line = line.strip().split('\t')
        lin_arr = []
        for i in range(21):
            lin_arr.append(float(cur_line[i]))
        train_set.append(lin_arr)
        train_labels.append(float(cur_line[21]))
    train_weights = stoc_grad_ascent(np.array(train_set), train_labels, NUM_ITERATORS)
    error_count = 0;num_test = 0
    for line in fr_test.readlines():
        num_test += 1
        cur_line = line.strip().split('\t')
        lin_arr = []
        for i in range(21):
            lin_arr.append(float(cur_line[i]))
        if (int(classify(lin_arr, train_weights)) != int(cur_line[21])):
            error_count += 1
    error_rate = float(error_count)/num_test
    return error_rate

def multi_test():
    num_test = 6
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colic_test()
    print("after %d iteration the average error rate is:%f" % (num_test, error_sum / float(num_test)))

def classify(sample, weights):
    prob = sigmoid(np.sum(sample*weights))
    if prob > 0.5:return 1.0
    else:return 0.0

def plot_best_fit(weights, filename):
    weights = weights.tolist()
    data_mat, label_mat = load_data_set(filename)
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(label_mat[i] == 1):
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == '__main__':
    multi_test()