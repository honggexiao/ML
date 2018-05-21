from numpy import *
import operator

def classify0(in_X, data_set, labels, k):
    data_size = data_set.shape[0]
    diff_mat = tile(in_X, (data_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    sq_dis = sq_diff_mat.sum(axis=1)
    dis = sq_dis**0.5
    sorted_dis = dis.argsort()
    #统计每个标签出现的频率
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dis[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

#生成简单数据集
def create_data_set():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0]
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#从文件打开数据集合
def file2matrix(filename):
    fr = open(filename, 'r', encoding='utf-8')
    array_lines = fr.readlines()
    num_lines = len(array_lines)
    data_mat = zeros((num_lines, 3))
    class_labels = []
    index = 0
    for line in array_lines:
        list_format_line = line.strip().split('\t')
        #这里有很多个特征，所以只取前三列作为分类的特征
        try:
            data_mat[index,:] = list_format_line[0:3]
            class_labels.append(int(list_format_line[-1]))
            index += 1
        except Exception as e:
            print('error data line: %d, error: %s' %(index + 1, str(e)))
    return data_mat, class_labels

def auto_norm(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_val, (m, 1))
    norm_data_set = data_set/tile(ranges, (m, 1))
    return norm_data_set, ranges, min_val