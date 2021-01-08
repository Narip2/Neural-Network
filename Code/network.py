import random
import network
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        # feedforward
        # x and y is list, not ndarray
        activation = x
        zs = [x]
        activations = [x]
        # contain activations
        for w, b in zip(self.weights, self.biases):
            activation = np.dot(w, activation)
            zs.append(activation)
            activation = sigmoid(activation + b)
            activations.append(activation)

        delta = (activation - y)*sigmoid_prime(zs[-1])
            # d_cost(activation, y)
        deltas = [delta]
        #compute delta
        #will not use the last layer
        zs.pop()
        for w, z in zip(reversed(self.weights),reversed(zs)):
            # print(delta.shape)
            # print(w.shape)
            # print(z.shape)
            delta = np.dot(w.transpose(), delta)*sigmoid_prime(z)
            deltas.append(delta)

        #delete the last one: no use
        deltas.pop()

        #calculate dc/db and dc/dw
        dws = []
        # print(deltas)
        # print(type(deltas))
        deltas = list(reversed(deltas))
        db = deltas
        for i in range(self.num_layers - 1):
            dw = np.dot(activations[i],list(deltas)[i].transpose())
            # print(deltas[i].shape)
            # print(activations[i].shape)
            # print(dw.shape)
            #need to use transpose()
            dws.append(dw.transpose())
        return db,dws

    #stochastic mini-batch gradient descent
    def SGD(self, tranning_data, epoch, mini_batch_size, eta, test_data = None):
        length = len(tranning_data)
        for i in range(epoch):
            random.shuffle(tranning_data)
            #leverage the fact that [a:b], b can exceed the length of the list
            mini_batchs = [tranning_data[k : k + mini_batch_size] for k in range(0,length,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.run(mini_batch, eta)
        #if have test_data, use evaluate to test
        if(test_data):
            num = self.evaluate(test_data)
            test_len = len(test_data)
            print(num/test_len)


    def run(self, training_data, eta):
        for x,y in training_data:
            db, dws = self.backprop(x,y)
            #update b and w
            temp = [eta*dwsone for dwsone in dws]
            self.weights = [weights - one for weights, one in zip(self.weights,temp)]
            temp = [eta*biasone for biasone in db]
            self.biases = [bias - one for bias, one in zip(self.biases,temp)]
            # self.weights = self.weights - eta*dws
            # self.biases = self.biases - [eta*arr for arr in db]


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #run one feedforward, do not change weights and biases, help evaluate function
    def feedforward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x


# cost = 1/2*(a - y)**2
def d_cost(self, a, y):
    return a - y


# active function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#derivitive of sigmoid
def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))
