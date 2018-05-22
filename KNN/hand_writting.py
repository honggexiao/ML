import os, sys
import numpy as np

from kNN import  file2matrix, classify0, auto_norm

TRAINING_DATA_DIR = 'trainingDigits'
TEST_DATA_DIR = 'testDigits'
K = 5

#将原始数据转换成一行向量,后期可用卷积神经网络CNN
def img2vector(filename):
    imag_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            imag_vec[0, i*32 + j] = int(line[j])
    return imag_vec

def handwritting_test():
    hw_labels = []
    training_file_list = os.listdir(TRAINING_DATA_DIR)
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    #处理训练数据
    for i in range(m):
        filename = training_file_list[i]
        file = filename.split('.')[0]
        num_class = int(file.split('_')[0])
        hw_labels.append(num_class)
        training_mat[i, :] = img2vector(TRAINING_DATA_DIR+'/'+filename)
    #处理测试数据
    test_file_list = os.listdir(TEST_DATA_DIR)
    error_count = 0
    m_test = len(test_file_list)
    #对每个测试集中的图像进行分类
    for i in range(m_test):
        filename = test_file_list[i]
        file = filename.split('.')[0]
        test_num_class = int(file.split('_')[0])
        test_image_vec = img2vector(TEST_DATA_DIR+'/'+filename)
        classifier_res = classify0(test_image_vec, training_mat, hw_labels, K)
        print("the classifier came back with: %d,the real answer is: %d" % (classifier_res, test_num_class))
        if (classifier_res != test_num_class):
            error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total rate is:%f" % (error_count / float(m_test)))

if __name__ == '__main__':
    handwritting_test()