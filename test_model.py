import numpy as np
import dnn
import dnnv2
from load_dataset import load_train_test_sets
import time

tic = time.time()
X_train, Y_train, X_test, Y_test = load_train_test_sets()
toc = time.time()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print("Time taken in loading data = ", (toc-tic)*1000, "ms")
n_x = int(256 * 256 * 3)
layer_dimensions = [n_x, 200, 70, 40, 1]
# 200, 70, 50, 1
tick = time.time()
parameters, costs = dnn.run_model(X_train,Y_train, layer_dimensions, learning_rate=0.75, num_iterations=20000, print_cost=True)
tock = time.time()

np.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning\Deep Neural network\dnn_learned_parameters", parameters)


print("Time taken in training model = ", (tock-tick)*1000, "ms")
#print(parameters)
print(costs)

"""
print("learned parameters--------------------------------------------------------------------------------")
learned_weights = np.load("cat-dog-classifier-dnn_weights.npy")
learned_bias = np.load("cat-dog-classifier-dnn_bias.npy")
print(learned_weights)
print(learned_bias)

"""