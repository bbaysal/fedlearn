import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self, layers_size):

        # layer_size parameter defines MLP dimensions. For example if layer_size [784, 50, 50, 10] then 
        # network has 3 layer and input dimension should be match with first value of array.
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)-1
        self.n = 0
        self.costs = []
        self.initialize_parameters()

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def initialize_parameters(self):
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

    def forward(self, X):
        store = {}

        A = X.T
        
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + \
                self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + \
            self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store
    
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def get_all_parameters(self, name):
        return self.parameters[name]

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter(self, name, value):
        self.parameters[name] = value
    
    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500, model_name=""):
        np.random.seed(1)

        self.n = X.shape[0]

        for loop in range(n_iterations):
            A, store = self.forward(X)

            cost = -np.mean(Y * np.log(A.T + 1e-8))

            derivatives = self.backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]

            if loop % 100 == 0:
                acc = self.predict(X, Y)
                if model_name != "":
                    print("Epoch: ", int(n_iterations/100), " / ",
                          int(loop/100), " --- Model:", model_name)
                else:
                    print("Epoch: ", int(n_iterations/100), " / ", int(loop/100))

                print("Model: ", model_name, " Cost: ",
                      cost, "Train Accuracy:", acc)

            if loop % 10 == 0:
                self.costs.append(cost)
    
    def predict(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()