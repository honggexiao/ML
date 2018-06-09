import numpy as np
import random

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1)for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data is not None:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #gradient descent to update weights and biases
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in  zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
                        gradient for the cost function C_x.  ``nabla_b`` and
                        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                        to ``self.biases`` and ``self.weights``."""
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backpropagation
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

if __name__ == '__main__':
    net = Network([2, 3, 1])
    print(net.weights)
    print(net.biases)