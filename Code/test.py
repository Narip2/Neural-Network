import network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)
# test_data = list(test_data)
# print(training_data)
net = network.Network([784,30,10])
net.SGD(training_data,30,10,3.0,test_data=test_data)
