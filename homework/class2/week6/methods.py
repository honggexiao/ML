import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
        Update parameters using one step of gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters to be updated:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients to update each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X= X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2
    V = {}
    for l in range(L):
        V['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        V['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
    return V

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2
    for l in range(L):
        v['dW'+str(l+1)] = beta*v['dW'+str(l+1)] + (1-beta)*grads['dW'+str(l+1)]
        v['db'+str(l+1)] = beta*v['db'+str(l+1)] + (1-beta)*grads['db'+str(l+1)]
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*v['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*v['db'+str(l+1)]
    return parameters, v


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        s['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        s['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        dW = 'dW'+str(l+1)
        db = 'db'+str(l+1)
        #update v ans s
        v[dW] = beta1*v[dW] + (1-beta1)*grads[dW]
        v[db] = beta1*v[db] + (1-beta1)*grads[db]
        v_corrected[dW] = v[dW]/(1 - beta1**t)
        v_corrected[db] = v[db]/(1 - beta1**t)

        s[dW] = beta2*s[dW] + (1-beta2)*grads[dW]**2
        s[db] = beta2*s[db] + (1-beta2)*grads[db]**2
        s_corrected[dW] = s[dW]/(1 - beta2**t)
        s_corrected[db] = s[db]/(1 - beta2**t)

        #update parameters
        W = 'W'+str(l+1)
        b = 'b'+str(l+1)
        parameters[W] = parameters[W] - learning_rate*(v_corrected[dW]/(np.sqrt(s_corrected[dW]) + epsilon))
        parameters[b] = parameters[b] - learning_rate*(v_corrected[db]/(np.sqrt(s_corrected[db]) + epsilon))

    return parameters, v, s


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    parameters = initialize_parameters(layers_dims)
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)
    for i in range(num_epochs):
        seed = seed + 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            a3, caches = forward_propagation(mini_batch_X, parameters)
            cost = compute_cost(a3, mini_batch_Y)
            grads = backward_propagation(mini_batch_X, mini_batch_Y, caches)
            #update parameters
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == 'adam':
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        if print_cost and i % 1000 == 0:
            print('Cost after epoch %i: %f'%(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title('learning rate= ' + str(learning_rate))
    plt.show()
    return parameters

if __name__ == '__main__':
    # parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    #
    # parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    # mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    #
    # print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    # print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    # print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    # print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    # print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    # print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    # print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
    # parameters = initialize_velocity_test_case()
    #
    # v = initialize_velocity(parameters)
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    # parameters, grads, v = update_parameters_with_momentum_test_case()
    #
    # parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    # parameters = initialize_adam_test_case()
    #
    # v, s = initialize_adam(parameters)
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    # print("s[\"dW1\"] = " + str(s["dW1"]))
    # print("s[\"db1\"] = " + str(s["db1"]))
    # print("s[\"dW2\"] = " + str(s["dW2"]))
    # print("s[\"db2\"] = " + str(s["db2"]))
    # parameters, grads, v, s = update_parameters_with_adam_test_case()
    # parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
    #
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    # print("s[\"dW1\"] = " + str(s["dW1"]))
    # print("s[\"db1\"] = " + str(s["db1"]))
    # print("s[\"dW2\"] = " + str(s["dW2"]))
    # print("s[\"db2\"] = " + str(s["db2"]))
    # train_X, train_Y = load_dataset()
    # layers_dims = [train_X.shape[0], 5, 2, 1]
    # parameters = model(train_X, train_Y, layers_dims, optimizer="gd")
    #
    # # Predict
    # predictions = predict(train_X, train_Y, parameters)
    #
    # # Plot decision boundary
    # plt.title("Model with Gradient Descent optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5, 2.5])
    # axes.set_ylim([-1, 1.5])
    # # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    # layers_dims = [train_X.shape[0], 5, 2, 1]
    # parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum")
    #
    # # Predict
    # predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    # plt.title("Model with Momentum optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5, 2.5])
    # axes.set_ylim([-1, 1.5])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    train_X, train_Y = load_dataset()
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)