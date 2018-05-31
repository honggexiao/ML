import numpy as np
import random

class optStruct:
    def __init__(self, data_mat_in, class_labels, C, toler, kTup=('lin', 0)):
        self.X = np.mat(data_mat_in)
        self.Y = np.mat(class_labels).transpose()
        self.C = C
        self.tol = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.E_cache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        #store kernal matrix
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i,:], kTup)

def load_data_set(filename):
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat

#calculate kernal with X and A
def kernel_trans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j,:] - A
            K[j] = delta_row*delta_row.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem')
    return K

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.Y).T*oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.Y[k])
    return Ek

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.E_cache[k] = [1, Ek]

def select_Jrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def select_j(i, oS, Ei):
    max_k = -1
    max_delta_E = 0
    Ej = 0
    oS.E_cache[i] = [1, Ei]
    valid_E_list = np.nonzero(oS.E_cache[:,0].A)[0]
    #选择误差最大的点为第二个变量
    if len(valid_E_list) > 1:
        for k in valid_E_list:
            if k == i:continue
            Ek = calcEk(oS, k)
            delta_E = np.abs(Ei - Ek)
            if delta_E > max_delta_E:
                max_k = k
                max_delta_E = delta_E
                Ej = Ek
        return max_k, Ej
    else:
        j = select_Jrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#inner_loop to choose second alpha
def inner_loop(i, oS):
    Ei = calcEk(oS, i)
    #kkt condition
    if (oS.Y[i]*Ei < -oS.tol and oS.alphas[i] < oS.C) or \
            (oS.Y[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = select_j(i, oS, Ei)
        alpha_i_old = oS.alphas[i].copy()
        alpha_j_old = oS.alphas[j].copy()
        if oS.Y[i] != oS.Y[j]:
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H:
            print('L==H')
            return 0
        eta = 2.0*oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        #update alpha[i] and alpha[j]
        oS.alphas[j] -= oS.Y[j]*(Ei - Ej)/eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.Y[i]*oS.Y[j]*(alpha_j_old - oS.alphas[j])
        updateEk(oS, i)
        #update b
        b1 = oS.b - Ei - oS.Y[i]*(oS.alphas[i] - alpha_i_old)*oS.K[i, i]-\
            oS.Y[j]*(oS.alphas[j] - alpha_j_old)*oS.K[i, j]
        b2 = oS.b - Ej - oS.Y[i]*(oS.alphas[i] - alpha_i_old) * oS.K[i, j] - \
             oS.Y[j] * (oS.alphas[j] - alpha_j_old) * oS.K[j, j]
        if 0 < oS.alphas[i] and oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2
        return 1
    else:
        return 0

def smo(data_mat_in, class_labels, C, toler, max_iter, kTup=('lin', 0)):
    oS = optStruct(data_mat_in, class_labels, C, toler, kTup)
    iter = 0
    entire_set = True
    alpha_pair_changed = 0
    while iter < max_iter and (alpha_pair_changed > 0 or entire_set):
        alpha_pair_changed = 0
        #outer loop to choose first alpha
        if entire_set:
            for i in range(oS.m):
                alpha_pair_changed += inner_loop(i, oS)
                print('fullSet, iter:%d i:%d, pairs changed %d' %(iter, i, alpha_pair_changed))
            iter += 1
        else:
            #choose support vector
            non_bounds = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<oS.C))[0]
            for i in non_bounds:
                alpha_pair_changed += inner_loop(i, oS)
                print('fullSet, iter:%d i:%d, pairs changed %d' %(iter, i, alpha_pair_changed))
            iter += 1
        if entire_set:entire_set = False
        elif alpha_pair_changed == 0:entire_set = True
        print(print ("iteration number: %d" % iter))
    return oS.b, oS.alphas

def calcWs(alphas, data_arr, class_labels):
    X = np.mat(data_arr)
    Y = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*Y[i], X[i].T)
    return w

if __name__ == '__main__':
    k1 = 0.1
    dataArr, labelArr = load_data_set('testSetRBF.txt')
    b, alphas = smo(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel_trans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = load_data_set('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernel_trans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))