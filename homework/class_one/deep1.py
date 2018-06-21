import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu_backward, relu

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
    params = {}
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    params['W1'] = W1
    params['W2'] = W2
    params['b1'] = b1
    params['b2'] = b2
    return params

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

#feedforward process
def linear_forward(A, W, b):
    cache = (A, W, b)
    Z = np.dot(W, A) + b
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        activation_fun = sigmoid
    else:
        activation_fun = relu
    A, cache = activation_fun(Z)
    activation_cache = Z
    cache = {'linear_cache':linear_cache, 'activation_cache':activation_cache}
    return A, cache

def L_model_forward(X, parameters):
    """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """
    L = len(parameters.items())//2
    caches = []
    A_prev = X.copy()
    for l in range(1, L):
        Wl = parameters['W'+str(l)]
        bl = parameters['b'+str(l)]
        A_prev, cache = linear_activation_forward(A_prev, Wl, bl, activation='relu')
        caches.append(cache)
    Al, cache = linear_activation_forward(A_prev, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)
    return Al, caches

def compute_cost(AL, Y):
    """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
    m = AL.shape[1]
    cost = 1./m*(-Y*np.log(AL)-(1.0 - Y)*np.log(1.0 - AL))
    cost = np.squeeze(cost)
    return cost

#back_propagation process
def linear_backward(dZ, cache):
    """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    m = dZ.shape[1]
    dW = np.dot(dZ, cache[0].transpose())/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].transpose(), dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    if activation == 'sigmoid':
        fun = sigmoid_backward
    else:
        fun = relu_backward
    Z = cache[1]
    dZ = fun(dA, Z)
    W = cache[0][1]
    m = Z.shape[1]
    dA_prev = np.dot(W.transpose(), dZ)
    dW = np.dot(dZ, cache[0][0].transpose())/m
    db = np.sum(dZ, axis=1)/m
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
    grads = {}
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    L = len(caches)
    cache = caches[L-1]
    dA, dW, db = linear_activation_backward(dAL, cache, 'sigmoid')
    grads['dA'+str(L)] = dAL
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    for i in range(L-2, -1, -1):
        grads['dA' + str(i + 1)] = dA
        cache = caches[i]
        dA, dW, db = linear_activation_backward(dA, cache, 'relu')
        grads['dW' + str(i + 1)] = dW
        grads['db' + str(i + 1)] = db
    return grads

def update_parameters(parameters, grads, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate*grads['d' + key]
    return parameters



if __name__ == '__main__':
    # params = initialize_parameters(3, 2, 1)
    # print("W1 = " + str(params["W1"]))
    # print("b1 = " + str(params["b1"]))
    # print("W2 = " + str(params["W2"]))
    # print("b2 = " + str(params["b2"]))
    # #Layer
    # parameters = initialize_parameters_deep([5, 4, 3])
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # A = np.random.randn(5, 3)
    # b = np.random.randn(4, 1)
    # z,cache = linear_forward(A, parameters['W1'], b)
    # print('z = ' + str(z))
    # A, W, b = linear_forward_test_case()
    #
    # Z, linear_cache = linear_forward(A, W, b)
    # print("Z = " + str(Z))
    # A_prev, W, b = linear_activation_forward_test_case()
    #
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    # print("With sigmoid: A = ",A)
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    # print("With ReLU: A = ", A)
    # X, parameters = L_model_forward_test_case_2hidden()
    # AL, caches = L_model_forward(X, parameters)
    # print("AL = " + str(AL))
    # print("Length of caches list = " + str(len(caches)))
    # dZ, linear_cache = linear_backward_test_case()
    #
    # dA_prev, dW, db = linear_backward(dZ, linear_cache)
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))
    # dAL, linear_activation_cache = linear_activation_backward_test_case()
    #
    # dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
    # print("sigmoid:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db) + "\n")
    #
    # dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
    # print("relu:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))
    # AL, Y_assess, caches = L_model_backward_test_case()
    # grads = L_model_backward(AL, Y_assess, caches)
    # print_grads(grads)
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))