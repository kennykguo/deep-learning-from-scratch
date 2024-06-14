import mnist_loader
training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
net.SGD(training_data , 5, 10, 3.0, test_data=test_data, save_file = "network_params.txt")