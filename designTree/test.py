import numpy as np
import matplotlib.pyplot as plt
import cart

if __name__ == '__main__':
    train_data = np.mat(cart.load_data_set('ex2.txt'))
    validate_data = np.mat(cart.load_data_set('ex2test.txt'))
    tree = cart.create_tree(train_data, ops=(0, 1),type='classify')
    tree = cart.prune(tree, validate_data)
    print(tree)
    val = cart.forcast_or_classify(tree, np.array([0.862030]))
    print(val)