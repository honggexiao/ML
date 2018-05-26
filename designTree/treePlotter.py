import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', fc=0.8)
leaf_node = dict(boxstyle='sawtooth', fc = 0.8)
arrow_args = dict(arrowstyle="<-")

#这是一个简单的算法，计算叶子节点的个数, 可以用广度优先搜索进行替换
def get_num_leafs(tree):
    num_leafs = 0
    first_str = list[tree][0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

#这也是一个很简单的算法，求一颗树的深度
def get_tree_depth(tree):
    max_depth = 0
    first_str = list[tree][0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(key).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth : max_depth = this_depth
    return max_depth