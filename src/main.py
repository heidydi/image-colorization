import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataset import Dataset
from neuralnet import Net
from trainer import Trainer
from loader import load
import matplotlib.pyplot as plt


def main(weights_file=None):
    window_size = 2
    (train_input, train_targets), (validation_input, validation_targets), (test_input, test_targets) = load(window_size)
    training_set = Dataset(train_input, train_targets, len(train_input))
    validation_set = Dataset(validation_input, validation_targets, len(validation_targets))
    testing_set = Dataset(test_input, test_targets, len(test_targets))
    print("Training data shape:", training_set.data.shape, training_set.targets.shape)
    print("Validation data shape:", validation_set.data.shape, validation_set.targets.shape)
    print("Testing data shape:", testing_set.data.shape, testing_set.targets.shape)
    # build the net
    input_dim = (2 * window_size + 1) ** 2
    hidden_dims = [8, 4]
    output_dim = 3
    lamb = 0.0001
    net = Net(input_dim, hidden_dims, output_dim, activation="sigmoid")
    net.printParams()
    batch_size = 1024
    sample_input, sample_targets = training_set.nextBatch(batch_size)
    assert net.checkGradient(sample_input, sample_targets, lamb) < 1e-10

    error_threshold = 1e-6
    agent = Trainer(error_threshold, batch_size, lamb)
    
    # If weights file is provided, load it and skip training
    if weights_file:
        if os.path.exists(weights_file):
            print(f"Loading weights from: {weights_file}")
            net.loadParams(weights_file)
        else:
            print(f"Error: Weights file not found at {weights_file}")
            return
    else:
        # train the neural net
        print("Training the net...")
        agent.train(net, training_set, validation_set)
        agent.evaluate(net, validation_set)
        print(net.layers[-1].outputs)
    
    # test store and load params
    print("Testing...")
    print(agent.evaluate(net, testing_set))
    print(net.layers[-1].outputs)
    from utils import array2pics
    from skimage import color
    import matplotlib.cm as cm
    original_pics = array2pics(test_targets, 32, 32)
    # gray_pics = color.rgb2gray(original_pics)
    predicted_pics = array2pics(net.layers[-1].outputs, 32, 32)
    _, axarr = plt.subplots(5, 8)
    for i in range(5):
        for j in range(3):
            axarr[i, 3*j].imshow(original_pics[i*10+j])
            axarr[i, 3*j+1].imshow(predicted_pics[i*10+j])
            axarr[i, 3*j].axis('off')
            axarr[i, 3*j+1].axis('off')
            if 3*j + 2 < 8:
                axarr[i, 3*j+2].axis('off')
    # Save to images directory in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'images', 'test_results.jpg')
    plt.savefig(output_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Colorization Model')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Optional weights filename to load and test the model (skips training)')
    args = parser.parse_args()
    main(weights_file=args.weights)
