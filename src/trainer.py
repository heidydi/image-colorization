import math
import numpy as np


class Trainer:
    def __init__(self, error_threshold, batch_size, lamb):
        self.error_threshold = error_threshold
        self.batch_size = batch_size
        self.lamb = lamb

    def train(self, net, training_set, validation_set):
        # if validation set stops improving for 'convergence_epochs', stop the training
        convergence_epochs = 10
        process_iter = 6
        # how many steps batch_min need to run for one epoch
        n = int(training_set.count / self.batch_size)
        epochs_count = 0
        # the parameter for ADAM update
        alpha = 0.001
        min_valid_error = self.evaluate(net, validation_set)
        training_errors = []
        validation_errors = []
        for i in range(process_iter):
            print("START PROCESS ITERATION:", i)
            net.resetAdam(alpha)  # reset alpha in ADAM
            alpha /= 10.
            catch_time = 0
            while catch_time < convergence_epochs:
                running_errors = 0.
                for _ in range(n):
                    input_data, output_data = training_set.nextBatch(self.batch_size)
                    running_errors += net.trainAdam(input_data, output_data, self.lamb)
                print("Average training error:", running_errors / n)
                training_errors.append(running_errors/n)
                validation_error = self.evaluate(net, validation_set)
                validation_errors.append(validation_error)
                if validation_error < min_valid_error:
                    print("Found better parameters at epochs:", epochs_count, " with validation error:", validation_error)
                    min_valid_error = validation_error
                    net.storeParams("_weights_")
                    catch_time = 0
                else:
                    catch_time += 1
                if min_valid_error < self.error_threshold:
                    print("Validation error smaller than threshold: validation error:", validation_error)
                    break
                epochs_count += 1
                if (epochs_count % 100) == 0:
                    net.storeParams("_weights_restart_")
                    self.storeErrors(training_errors, validation_errors)

    def storeErrors(self, training_errors, validation_errors):
        np.savetxt("_training_errors_", training_errors)
        np.savetxt("_validation_errors_", validation_errors)

    def evaluate(self, net, validation_set):
        # ret = 0
        # while validation_set.pos != validation_set.count:
        #     if validation_set.pos + self.batch_size > validation_set.count:
        #         input_data, output_data = validation_set.nextBatch(validation_set.count - validation_set.pos)
        #     else:
        #         input_data, output_data = validation_set.nextBatch(self.batch_size)
        #     ret += net.evaluate(input_data, output_data)
        # validation_set.pos = 0
        # return math.sqrt(ret / validation_set.count)
        return net.squareCost(validation_set.data, validation_set.targets, self.lamb)
