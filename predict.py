# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
#import cv2
#import time
#start_time = time.time()

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    #warnings.filterwarnings('ignore')
    data = load_data()
    X_train = data[0]
    Y_train = data[1]
    length = len(X_train)
    cut_off = length * 0.18
    X_train = X_train[0: cut_off, 0:cut_off]
    Y_train = Y_train[0: cut_off, :]
    #dist = hamming_distance(x[:, 0:cut_off], X_train)
    #return sort_train_labels_knn(dist, Y_train)
    pass

def load_data():
    FILE_PATH = 'train.pkl'
    with open(FILE_PATH, 'rb') as f:
        return pkl.load(f)


from numpy import exp, array, random, dot

data = load_data()
X_train = data[0]
Y_train = data[1]
length = len(X_train)
cut_off = length * 0.10
X_train = X_train[0: cut_off, 0:cut_off]
Y_train = Y_train[0: cut_off, :]

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights =35 * random.random((cut_off,1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    #training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    #training_set_outputs = array([[0, 1, 1, 0]]).T

    training_set_inputs = X_train
    training_set_outputs = Y_train
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10)
    #neural_network.train(X_train, Y_train, 10)
    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print ("Considering new situation ?: ")
    print(X_train[5])
    print(Y_train[0])
    print (neural_network.think(X_train[10]))

    """
    network.py
    ~~~~~~~~~~
    A module to implement the stochastic gradient descent learning
    algorithm for a feedforward neural network.  Gradients are calculated
    using backpropagation.  Note that I have focused on making the code
    simple, easily readable, and easily modifiable.  It is not optimized,
    and omits many desirable features.
    """

    #### Libraries
    # Standard library
    import random

    # Third-party libraries
    import numpy as np


    class Network(object):

        def __init__(self, sizes):
            """The list ``sizes`` contains the number of neurons in the
            respective layers of the network.  For example, if the list
            was [2, 3, 1] then it would be a three-layer network, with the
            first layer containing 2 neurons, the second layer 3 neurons,
            and the third layer 1 neuron.  The biases and weights for the
            network are initialized randomly, using a Gaussian
            distribution with mean 0, and variance 1.  Note that the first
            layer is assumed to be an input layer, and by convention we
            won't set any biases for those neurons, since biases are only
            ever used in computing the outputs from later layers."""
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]

        def feedforward(self, a):
            """Return the output of the network if ``a`` is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a

        def SGD(self, training_data, epochs, mini_batch_size, eta,
                test_data=None):
            """Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
            if test_data: n_test = len(test_data)
            n = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k + mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    print(mini_batch)
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print
                    "Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test)
                else:
                    print
                    "Epoch {0} complete".format(j)

        def update_mini_batch(self, mini_batch, eta):
            """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]

        def backprop(self, x, y):
            """Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x]  # list to store all the activations, layer by layer
            zs = []  # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * \
                    sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return (nabla_b, nabla_w)

        def evaluate(self, test_data):
            """Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation."""
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in test_data]
            return sum(int(x == y) for (x, y) in test_results)

        def cost_derivative(self, output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            return (output_activations - y)


    #### Miscellaneous functions
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z) * (1 - sigmoid(z))




    net = Network([784, 30, 10])

    net.SGD(X_train, 30, 10, 3.0, )