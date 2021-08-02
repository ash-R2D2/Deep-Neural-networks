import numpy as np
import dnn
from load_dataset import load_train_test_sets
import time


def train_model():
    tic = time.time()
    X_train, Y_train, X_test, Y_test = load_train_test_sets()
    toc = time.time()

    print("Time taken in loading data = ", (toc - tic) * 1000, "ms")
    n_x = int(64 * 64 * 3)
    layer_dimensions = [n_x, 300, 170, 60, 1]
    # 200, 70, 50, 1

    tick = time.time()
    parameters, costs = dnn.run_model(X_train, Y_train, layer_dimensions, learning_rate=0.015, num_iterations=13500,
                                      print_cost=True)
    tock = time.time()

    np.save(r"C:\Users\avina\Documents\GitHub\Deep-Neural-networks\cat_v_dog_dnn_learned_parameters_v1.1", parameters)
    print("Time taken in training model = ", (tock - tick) * 1000, "ms")

    return 0


def test_accuracy(parameters):
    X_train, Y_train, X_test, Y_test = load_train_test_sets()
    print(" Dataset Loaded Successfully!!!......")

    Y_train_predictions = dnn.predict(X_train, parameters)
    Y_test_predictions = dnn.predict(X_test, parameters)
    print("Predictions computed successfully!!!....")

    train_set_accuracy = (100 - np.mean(np.abs(Y_train_predictions - Y_train)) * 100)
    test_set_accuracy = (100 - np.mean(np.abs(Y_test_predictions - Y_test)) * 100)

    print("Accuracy on training set is = ", train_set_accuracy, "%")
    print("Accuracy on test set is = ", test_set_accuracy, "%")

    return 0


parameters = np.load(r"C:\Users\avina\Documents\GitHub\Deep-Neural-networks\cat_v_dog_dnn_learned_parameters_v1.1.npy",
                     allow_pickle=True).flat[0]

test_accuracy(parameters)
