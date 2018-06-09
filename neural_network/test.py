import network
import mnist_loader

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    print(len(test_data))
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 400, 10, 3.0, test_data=test_data)
