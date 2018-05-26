import numpy as np
import operator
import collections
import copy

leaf = collections.namedtuple('leaf', 'val class_cnt')

def load_data_set(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat

def bin_split_data_set(data_set, feature, value, type='regress'):
    """根据类型type进行划分"""
    if type == 'regress':
        mat0 = data_set[np.nonzero(data_set[:,feature] > value)[0], :]
        mat1 = data_set[np.nonzero(data_set[:,feature] <= value)[0], :]
    else:
        mat0 = data_set[np.nonzero(data_set[:, feature] == value)[0], :]
        mat1 = data_set[np.nonzero(data_set[:, feature] != value)[0], :]
    return mat0, mat1

#计算特征的均值
def reg_leaf(data_set):
    return leaf(np.mean(data_set[:, -1]), None)
#计算目标的总方差，均方误差*总样本数
def reg_error(data_set):
    return np.var(data_set[:, -1])*np.shape(data_set)[0]
#根据多数表决进行分类
def classify_leaf(data_set):
    class_count = {}
    for val in data_set[:,-1].T.tolist[0]:
        class_count[val] = class_count.get(val, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return leaf(sorted_class_count[0][0], class_count)
#计算基尼指数
def classify_error(data_set):
    num_entries = data_set.shape[0]
    class_count = {}
    for val in data_set[:, -1].T.tolist[0]:
        class_count[val] = class_count.get(val, 0) + 1
    gini = 1.0
    for _, val in class_count.items():
        gini -= np.power(float(val/num_entries), 2)
    return gini

#这里可以根据多种规格进行最优特征选择，自己实现一种基于Gini系数的分类树构造方法
#ops控制着停止条件，ops[0]控制允许误差的下降值，ops[1]控制切分的最小样本数
def choose_best_split(data_set, leaf_type=reg_leaf, err_type = reg_error, ops=(1, 4), type='regress'):
    tols = ops[0];toln = ops[1]
    if len(set(data_set[:,-1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m,n = np.shape(data_set)
    #计算总方差
    s = err_type(data_set)
    best_s = np.inf;best_index = 0;best_value = 0
    for feat_index in range(n - 1):
        for split_val in set((data_set[:,feat_index].T.A.tolist())[0]):
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val, type)
            #样本数小于给定阈值不允许进行划分, 实际上这里是一种预剪枝操作
            if np.shape(mat0)[0] < toln or np.shape(mat1)[0] < toln:
                continue
            coeff0 = 1.0;coeff1 = 1.0
            if type == 'classify':
                coeff0 = float(np.shape(mat0)[0]/m)
                coeff1 = float(np.shape(mat1)[0]/m)
            new_s = coeff0*err_type(mat0) + coeff1*err_type(mat1)
            if new_s < best_s:
                best_s = new_s
                best_index = feat_index
                best_value = split_val
    #如果误差减小不大，则停止切分
    if (s - best_s) < tols:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value, type)
    #这一步是必须的步骤，并非多此一举，假设这里不加这个判断，那么如果每个特征都小于最小
    #切分样本数，那么这里返回的就是0，实际上还是进行了切分，但是加了，就防止了这种情况
    if np.shape(mat0)[0] < toln or np.shape(mat1)[0] < toln:
        return None, leaf_type(data_set)
    return best_index, best_value

def create_tree(data_set, leaf_type = reg_leaf, err_type=reg_error, ops=(1, 4), type='regress'):
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops, type)
    #达到叶子节点，可以根据停止准则自己添加代码，后续我会添加新的准则，Gini系数小于某个阈值
    if feat == None: return val
    ret_tree = {}
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    l_set, r_set = bin_split_data_set(data_set, feat, val, type)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops, type)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops, type)
    return ret_tree

#merge two leaf node
def merge_leaf(lchild, rchild):
    if lchild is None: return rchild
    if rchild is None: return lchild
    #if task is classify
    if lchild.class_cnt is not None:
        ret_class_cnt = copy.deepcopy(lchild.class_cnt)
        best_class = lchild.val
        best_cnt = lchild.class_cnt[best_class]
        if rchild.class_cnt[rchild.val] > best_cnt:
            best_class = rchild.val
            best_cnt = rchild.class_cnt[rchild.val]
        for key, val in rchild.class_cnt.items():
            if key in ret_class_cnt:
                cnt_gain = ret_class_cnt.get(key, 0) + val
                if cnt_gain > best_cnt:
                    best_cnt = cnt_gain
                    best_class = key
                    ret_class_cnt[key] = cnt_gain
        return leaf(best_class, ret_class_cnt)
    else: return leaf((lchild.val + rchild.val)/2.0, None)

def is_tree(obj):
    return type(obj).__name__ == 'dict'

#利用验证数据对tree进行递归剪枝
def prune(tree, test_data, type='regress'):
    # if np.shape(test_data)[0] == 0:return get_mean(tree)
    if (is_tree(tree['right']) or is_tree(tree['left'])):
        l_set, r_set = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'], type)
        if is_tree(tree['left']):
                tree['left'] = prune(tree['left'], l_set, type)
        if is_tree(tree['right']):
            tree['right'] = prune(tree['right'], r_set, type)
    #这里必须先进行剪枝处理，否则只会对叶子节点进行合并，这是一个向上冒泡的过程
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_set, r_set = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'], type)
        if type == 'regress':
            error_no_merge = np.sum(np.power(l_set[:, -1] - tree['left'].val, 2)) + \
                np.sum(np.power(r_set[:, -1] - tree['right'].val, 2))
            tree_mean = (tree['left'].val + tree['right'].val) / 2.0
            error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        else:
            num_entires = np.shape(test_data)[0]
            error_merge = classify_error(test_data)
            error_no_merge = float(np.shape(l_set)[0]/num_entires)*classify_error(l_set)+ \
                             float(np.shape(r_set)[0]/num_entires)*classify_error(r_set)
        #回归树利用总方差对是否进行剪枝进行判断
        if error_merge < error_no_merge:
            print('merging')
            return merge_leaf(tree['left'], tree['right'])
        else:return tree
    else:
        return tree

#针对回归模型进行预测
def forcast_or_classify(tree, sample, type='regress'):
    if not is_tree(tree):
        return tree.val
    if type == 'classify':
        if sample[tree['sp_ind']] == tree['sp_val']:
            return forcast_or_classify(tree['left'], sample)
        else:
            return forcast_or_classify(tree['right'], sample)
    if sample[tree['sp_ind']] <= tree['sp_val']:
        return forcast_or_classify(tree['right'], sample)
    else: return forcast_or_classify(tree['left'], sample)

if __name__ == '__main__':
    pass