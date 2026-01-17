import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return 0 if x <= 0 else x


def relu_derivative(x):
    return 0 if x <= 0 else 1


def tanh(x):
    return 2. / (1. + np.exp(-2 * x)) - 1


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


class Adam:
    def __init__(self):
        self.t = 0
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def update(self):
        self.t += 1

    def reset(self, alpha):
        self.t = 0
        self.alpha = alpha


class Layer:
    def __init__(self, inputcount, neuroncount, fan_out):
        self.inputcount = inputcount
        self.neuroncount = neuroncount
        self.fan_out = fan_out

        self.outputs = None

        self.weights = np.zeros([self.inputcount, self.neuroncount])
        self.bias = np.zeros(self.neuroncount)
        self.weights_grad = np.zeros([self.inputcount, self.neuroncount])
        self.bias_grad = np.zeros(self.neuroncount)
        self.init_weights()

        self.adam_weightsm = np.zeros([self.inputcount, self.neuroncount])
        self.adam_weightsv = np.zeros([self.inputcount, self.neuroncount])
        self.adam_biasm = np.zeros(self.neuroncount)
        self.adam_biasv = np.zeros(self.neuroncount)

    def init_weights(self):
        xavier_limit = math.sqrt(6.) / math.sqrt(self.inputcount + self.fan_out + 1.)
        self.weights = (np.random.rand(self.inputcount, self.neuroncount) - 0.5) * 2 * xavier_limit

    def update_weights(self, adam):
        alphat = adam.alpha * math.sqrt(1 - math.pow(adam.beta2, adam.t)) / (1 - math.pow(adam.beta1, adam.t))
        self.adam_weightsm = adam.beta1 * self.adam_weightsm + (1. - adam.beta1) * self.weights_grad
        self.adam_weightsv = adam.beta2 * self.adam_weightsv + (1. - adam.beta2) * (self.weights_grad ** 2)
        self.weights -= alphat * self.adam_weightsm / (np.sqrt(self.adam_weightsv) + adam.epsilon)

        self.adam_biasm = adam.beta1 * self.adam_biasm + (1. - adam.beta1) * self.bias_grad
        self.adam_biasv = adam.beta2 * self.adam_biasv + (1. - adam.beta2) * (self.bias_grad ** 2)
        self.bias -= alphat * self.adam_biasm / (np.sqrt(self.adam_biasv) + adam.epsilon)

    def reset(self):
        self.adam_weightsm = np.zeros([self.inputcount, self.neuroncount])
        self.adam_weightsv = np.zeros([self.inputcount, self.neuroncount])
        self.adam_biasm = np.zeros(self.neuroncount)
        self.adam_biasv = np.zeros(self.neuroncount)


class Net:
    def __init__(self, input_dim, hidden_dims, output_dim, activation=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layer_count = len(hidden_dims) + 1

        # init layers
        self.layers = []
        structure = self.build_structure()
        for i in range(self.layer_count):
            self.layers.append(Layer(structure[i], structure[i+1], structure[i+2]))
        # init adam params
        self.adam = Adam()
        # set up activation function
        if activation is None or activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivation = sigmoid_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivation = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivation = tanh_derivative
        else:
            print("Unknown activation function:", activation)

    def trainAdam(self, input_data, targets, lamb):
        self.adam.update()
        self.propagate(input_data)
        errort = self.backprop(input_data, targets, lamb)
        self.updateWeights()
        return errort

    def propagate(self, input_data):
        batch_size = len(input_data)
        for layer in self.layers:
            # copy the bias to outputs
            layer.outputs = np.zeros([batch_size, layer.neuroncount])
            for i in range(batch_size):
                layer.outputs[i] = layer.bias
            # compute the weights multiplied by input
            layer.outputs += np.dot(input_data, layer.weights)
            # apply activation !!! Note: the last layer is also applied with activation function TODO?
            layer.outputs = self.activation(layer.outputs)
            # update input
            input_data = layer.outputs

    def backprop(self, input_data, targets, lamb):
        batch_size = len(input_data)
        if batch_size == 0:
            print("input data size 0!")
            return
        # compute ouput error
        errorvalues = self.layers[-1].outputs - targets
        errort = np.sum(errorvalues ** 2) / (2 * batch_size)
        # compute the gradients
        errorvalues *= self.activation_derivation(self.layers[-1].outputs)
        for i in np.arange(self.layer_count-1, -1, -1):
            layerinput = input_data if i == 0 else self.layers[i-1].outputs
            # errorvalues * layerinput -> gradient
            self.layers[i].weights_grad = np.dot(layerinput.T, errorvalues)
            # gradient for bias
            self.layers[i].bias_grad = np.dot(np.ones(batch_size), errorvalues)
            # update errorvalues
            if i > 0:
                errorvalues = np.dot(errorvalues, self.layers[i].weights.T) * self.activation_derivation(self.layers[i-1].outputs)
        # add the weight decay term and average the gradient
        for layer in self.layers:
            layer.weights_grad *= 1. / batch_size
            layer.weights_grad += lamb * layer.weights / batch_size
            layer.bias_grad *= 1. / batch_size
        # return the average error
        return errort + self.weightDecay(lamb, batch_size)

    def updateWeights(self):
        for layer in self.layers:
            layer.update_weights(self.adam)

    # def evaluate(self, input_data, targets, lamb):
    #     return self.squareCost(input_data, targets, lamb)

    def weightDecay(self, lamb, batch_size):
        weight_decay = 0
        for layer in self.layers:
            weight_decay += np.sum(layer.weights ** 2)
        return lamb * weight_decay / (2 * batch_size)

    def build_structure(self):
        structure = []
        structure.append(self.input_dim)
        for item in self.hidden_dims:
            structure.append(item)
        structure.append(self.output_dim)
        structure.append(1)
        return structure

    def checkGradient(self, input_data, targets, lamb):
        """compute the numerically computed gradients and back propagation computed gradients"""
        gradient_numerical = self.computeNumericalGradient(input_data, targets, lamb)

        self.propagate(input_data)
        self.backprop(input_data, targets, lamb)
        gradient_bp = np.zeros(len(gradient_numerical))
        idx = 0
        for layer in self.layers:
            for i in range(layer.inputcount):
                for j in range(layer.neuroncount):
                    gradient_bp[idx] = layer.weights_grad[i][j]
                    idx += 1
            for i in range(layer.neuroncount):
                gradient_bp[idx] = layer.bias_grad[i]
                idx += 1
        print(gradient_numerical)
        print(gradient_bp)
        # compute norm1 / norm2
        norm1 = (gradient_bp - gradient_numerical) ** 2
        norm2 = (gradient_bp + gradient_numerical) ** 2
        norm = np.sum(norm1) / np.sum(norm2)
        print("Norm:", norm)
        return norm

    def computeNumericalGradient(self, input_data, targets, lamb):
        params_size = 0
        for layer in self.layers:
            params_size += (layer.inputcount + 1) * layer.neuroncount
        gradient_numerical = np.zeros(params_size)

        idx = 0
        epsilon = 0.0001
        for layer in self.layers:
            # go over the weights
            for i in range(layer.inputcount):
                for j in range(layer.neuroncount):
                    layer.weights[i][j] += epsilon
                    j1 = self.squareCost(input_data, targets, lamb)
                    layer.weights[i][j] -= 2 * epsilon
                    j2 = self.squareCost(input_data, targets, lamb)
                    layer.weights[i][j] += epsilon
                    gradient_numerical[idx] = (j1 - j2) / (2 * epsilon)
                    idx += 1
            # go over the bias
            for i in range(layer.neuroncount):
                layer.bias[i] += epsilon
                j1 = self.squareCost(input_data, targets, lamb)
                layer.bias[i] -= 2 * epsilon
                j2 = self.squareCost(input_data, targets, lamb)
                layer.bias[i] += epsilon
                gradient_numerical[idx] = (j1 - j2) / (2 * epsilon)
                idx += 1

        return gradient_numerical

    def squareCost(self, input_data, targets, lamb):
        batch_size = len(input_data)
        self.propagate(input_data)
        errort = np.sum((self.layers[-1].outputs - targets) ** 2) / (2. * batch_size)

        return errort + self.weightDecay(lamb, batch_size)

    def loadParams(self, filename):
        params = np.loadtxt(filename)
        idx = 0
        for layer in self.layers:
            for i in range(layer.inputcount):
                for j in range(layer.neuroncount):
                    layer.weights[i][j] = params[idx]
                    idx += 1
            for i in range(layer.neuroncount):
                layer.bias[i] = params[idx]
                idx += 1
        return params

    def storeParams(self, filename):
        params_size = 0
        for layer in self.layers:
            params_size += (layer.inputcount + 1) * layer.neuroncount
        params = np.zeros(params_size)

        idx = 0
        for layer in self.layers:
            for i in range(layer.inputcount):
                for j in range(layer.neuroncount):
                    params[idx] = layer.weights[i][j]
                    idx += 1
            for i in range(layer.neuroncount):
                params[idx] = layer.bias[i]
                idx += 1

        np.savetxt(filename, params)

    def resetAdam(self, alpha):
        self.adam.reset(alpha)
        for layer in self.layers:
            layer.reset()

    def printParams(self):
        idx = 0
        for layer in self.layers:
            print("Layer", idx)
            print("Weights:")
            print(layer.weights)
            print("Bias:")
            print(layer.bias)
            idx += 1
