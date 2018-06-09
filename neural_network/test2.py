import network2
import mnist_loader


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    print(len(test_data))
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, lmbda = 5, evaluation_data=validation_data , monitor_evaluation_accuracy=True, monitor_training_cost=True)
