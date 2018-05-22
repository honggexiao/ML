from numpy import *

def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱类，0表示不属于
    return posting_list, class_vec
def create_vocab_list(documents):
    vocab_set = set([])
    for document in documents:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def create_words_vec(vocab_list, document):
    """对每篇文档建立词汇表"""
    word_vec = [0]*len(vocab_list)
    for word in document:
        if word in vocab_list:
            word_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return word_vec

def train_bayes(train_mat, train_cat):
    num_train_docs = len(train_mat)
    #单词表的长度
    num_words = len(train_mat[0])
    p_abusive = sum(train_cat)/float(num_train_docs)
    p_num_0 = zeros(num_words)
    p_num_1 = zeros(num_words)
    p_denom_0 = 0.0
    p_denom_1 = 0.0
    for i in range(num_train_docs):
        if train_cat[i] == 1:
            p_num_1 += train_mat[i]
            p_denom_1 += sum(train_mat[i])
        else:
            p_num_0 += train_mat[i]
            p_denom_0 += sum(train_mat[i])
    p_vec_0 = log(p_num_1/p_denom_1)
    p_vec_1 = log(p_num_0/p_denom_0)
    return p_vec_0, p_vec_1, p_abusive

def classify_bayes(vec_classify, p_vec_0, p_vec_1, p_class_1):
    p1 = log(p_class_1) + sum(vec_classify*p_vec_1)
    p0 = log(1.0 - p_class_1) + sum(vec_classify*p_vec_0)
    if p1 > p0:
        return 1
    else:
        return 0