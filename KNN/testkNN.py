import matplotlib.pyplot as plt
import matplotlib
from numpy import *

from kNN import  file2matrix, classify0, auto_norm

DATING_DATA = 'datingTestSet2.txt'
HO_RATIO = 0.1
K = 5

#约会网站的测试
def dating_class_test():
    dating_data_mat, dating_labels = file2matrix(DATING_DATA)
    norm_data, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_data.shape[0]
    num_test_vecs = int(m*HO_RATIO)
    error_count = 0.0
    for i in  range(num_test_vecs):
        classifier_res = classify0(norm_data[i, :], norm_data[num_test_vecs:m,:], dating_labels[num_test_vecs:m], K)
        print('the classifier came back with: %d, the real answer is: %d'%(classifier_res, dating_labels[i]))
        if (classifier_res != dating_labels[i]): error_count += 1.0
    print("the total error rate is: %f" % (error_count/float(num_test_vecs)))

def feature_show():
    dating_data_mat, dating_labels = file2matrix(DATING_DATA)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
    bx = fig.add_subplot(122)
    bx.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()

if __name__ == '__main__':
    feature_show()
    dating_class_test()