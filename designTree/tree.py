from math import log
import operator
import pickle

#计算香农熵
def cal_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for fear_vedc in data_set:
        curr_label = fear_vedc[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]/num_entries)
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent

#value要表示划分的值,划分数据的时候同时删除这个特征
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

def choose_best_feature_to_split(data_set):
    #最后一个为类别，所以此时需要减去1
    num_features = len(data_set[0]) - 1
    base_entropy = cal_shannon_ent(data_set)
    best_info_gain = 0.0; best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob*cal_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    #判断是否是叶子节点，如果是则返回叶子节点所表示的类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    #选择最优划分属性
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    #以字典来存储一颗决策树
    my_tree = {best_feat_label:{}}
    del(labels[best_feat])
    feat_vals = [example[best_feat] for example in data_set]
    unique_vals = set(feat_vals)
    #递归创建决策树
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree

def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree)[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels[first_str]
    key = test_vec[feat_index]
    val_of_feat = second_dict[key]
    if isinstance(val_of_feat, dict):
        #递归的进行寻找
        class_label = classify(val_of_feat, feat_labels,  test_vec)
    else:
        class_label = val_of_feat
    return class_label

def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def store_tree(tree, filename):
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()

def grab_tree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
   fr = open('lenses.txt')
   lenses = [inst.strip().split('\t') for inst in fr.readlines()]
   lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
   lenses_tree = create_tree(lenses, lenses_labels)
   print(lenses_tree)
