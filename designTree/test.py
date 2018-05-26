import numpy as np
import matplotlib.pyplot as plt
import cart

if __name__ == '__main__':
    # mydat = np.mat(cart.load_data_set('ex00.txt'))
    # print(mydat[:, -1].T.tolist()[0])
    # tree = cart.create_reg_tree(mydat)
    # print(tree)
    # plt.plot(mydat[:,0], mydat[:,1], 'ro')
    # plt.show()
    # mydat1 = np.mat(cart.load_data_set('ex0.txt'))
    # tree1 = cart.create_reg_tree(mydat1)
    # print(tree1)
    # plt.plot(mydat1[:, 1], mydat1[:, 2], 'ro')
    # plt.show()
    train_data = np.mat(cart.load_data_set('ex2.txt'))
    validate_data = np.mat(cart.load_data_set('ex2test.txt'))
    tree = cart.create_tree(train_data, ops=(0, 1),type='classify')
    tree = cart.prune(tree, validate_data)
    print(tree)
    val = cart.forcast_or_classify(tree, np.array([0.862030]))
    print(val)