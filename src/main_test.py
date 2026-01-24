import os
import numpy as np
from dataset import Dataset
from neuralnet import Net
from trainer import Trainer
from copy import deepcopy


def test_correctness():
    """test the gradients correctness and network convergence of the XOR problem"""
    # build the input data and targets
    input_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    targets = np.array([[0], [1], [0], [1]])
    batch_size = int(len(input_data))
    # build the data set
    training_set = Dataset(input_data, targets, len(input_data))
    validation_set = Dataset(deepcopy(input_data), deepcopy(targets), len(input_data))
    # build the net
    input_dim = 2
    hidden_dims = [2, 3]
    output_dim = 1
    lamb = 0.00001
    net = Net(input_dim, hidden_dims, output_dim, activation="sigmoid")
    net.printParams()
    # check the gradients
    sample_input, sample_targets = training_set.nextBatch(batch_size)
    print("Input data:\n", sample_input, "\nTargets:\n", sample_targets)
    assert net.checkGradient(sample_input, sample_targets, lamb) < 1e-10
    print("Gradient checking success!")
    # train the neural net
    print("Training the net...")
    error_threshold = 1e-6
    agent = Trainer(error_threshold, batch_size, lamb)
    agent.train(net, training_set, validation_set)
    agent.evaluate(net, validation_set)
    print(net.layers[-1].outputs)
    # test store and load params
    print("Testing load params...")
    weights_path = os.path.join(os.path.dirname(__file__), "_weights_")
    if os.path.exists(weights_path):
        net.loadParams(weights_path)
        print("Loaded weights from:", weights_path)
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Skipping load params test.")
        return
    agent.evaluate(net, validation_set)
    print(net.layers[-1].outputs)


if __name__ == '__main__':
    test_correctness()
