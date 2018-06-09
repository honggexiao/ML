import numpy as np

def load_simple_data():
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, class_labels

def load_data_set(filename):
    num_feat = len(open(filename).readline().split('\t'))
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        lin_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            lin_arr.append(float(cur_line[i]))
        data_mat.append(lin_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat

def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:,dimen] <= thresh_val] = -1.0
        ret_array[data_matrix[:,dimen] > thresh_val] = 1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array

#build decision stump
def build_stump(data_arr, class_labels, D):
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10
    best_stump = {}
    best_class_set = np.mat(np.zeros((m,1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_mat[:,i].min()
        range_max = data_mat[:,i].max()
        step_size = (range_max - range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in  ['lt', 'gt']:
                thresh_val = range_min + step_size*float(j)
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m,1)))
                err_arr[predicted_vals==label_mat]=0
                weight_error = float(D.T*err_arr)
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_set = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_set

def add_boost(data_arr, class_labels, num_iter=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1))/m)
    agg_class_set = np.mat(np.zeros((m,1)))
    for i in range(num_iter):
        best_stump, min_error, best_class_set = build_stump(data_arr, class_labels, D)
        #compute parameter alpha
        alpha = float(0.5*np.log((1-min_error)/max(min_error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        #compute weight array
        expon = np.multiply(-1*alpha*np.mat(class_labels).T, best_class_set)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        agg_class_set += alpha*best_class_set
        agg_errors = np.multiply(np.sign(agg_class_set) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print("total error: ", error_rate)
        if error_rate == 0.0: break
    return weak_class_arr

def add_classify(samples, classifiers):
    data_mat = np.mat(samples)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifiers)):
        class_est = stump_classify(data_mat, classifiers[i]['dim'], classifiers[i]['thresh'], classifiers[i]['ineq'])
        agg_class_est += classifiers[i]['alpha']*class_est
    return np.sign(agg_class_est)

if __name__ == '__main__':
    train_data, train_targets = load_data_set('horseColicTraining2.txt')
    classifiers = add_boost(train_data, train_targets, 50)
    #print(classifiers)
    test_data, test_labels = load_data_set('horseColicTest2.txt')
    targets = add_classify(test_data, classifiers)
    print(targets)
    err_mat = np.mat(np.ones((67, 1)))
    err_rate = err_mat[targets != np.mat(test_labels).T].sum()/67
    print(err_rate)
    # D = np.mat(np.ones((5, 1))/5)
    # best_stump, min_error, best_class_set = build_stump(data_mat, class_labels, D)
    # print(best_stump, min_error, best_class_set)