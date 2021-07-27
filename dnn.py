import numpy as np
from dnn_utils import sigmoid, relu, sigmoid_backward, relu_backward
from numba import jit, cuda


def initialize_parameters(layer_dimensions):
    parameters = {}

    # layer_dimensions[0] is the number of input features in a training example or --
    # the number of neurons in the training layer.

    # Total number of layers including the input layer.
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
        parameters["B" + str(l)] = np.zeros((layer_dimensions[l], 1))

        #assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
        #assert (parameters['B' + str(l)].shape == (layer_dimensions[l], 1))


    return parameters

# 
def linear_hypothesis(A, W, B):
    # Here A is the activation of the previous layer and is acting as the
    # input for the current layer. W and B are the parameters of the current layer.
    Z = np.dot(W, A) + B
    cache = (A, W, B)
    return Z, cache

# 
def hypothesis_activation(A_prev, W, B, activation_fn):
    Z, linear_cache = linear_hypothesis(A_prev, W, B)
    # The linear_cache contains the A (A_prev), W , B used to calculate the Z.
    # Here the A_prev is the activation of previous layer which is acting as the input for the current layer
    # and is being used with the W and B of current layers to calculate the Z of the current layer.

    if activation_fn == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "relu":
        A, activation_cache = relu(Z)

    # The activation_cache contains the value of the Z which is used to calculate the activation A
    # using the given function.

    cache = (linear_cache, activation_cache)

    return A, cache

# 
def forward_propagation(X, parameters):
    forward_cache = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = hypothesis_activation(A_prev, parameters["W" + str(l)], parameters["B" + str(l)], "relu")
        forward_cache.append(cache)

    AL, cache = hypothesis_activation(A, parameters["W" + str(L)], parameters["B" + str(L)], "sigmoid")
    forward_cache.append(cache)

    return AL, forward_cache

# 
def compute_cost(AL, Y):
    m = Y.shape[1]

    # Computing the cross entropy loss
    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))
    cost = np.squeeze(cost)

    return cost

# 
def linear_backward(dZ, cache):
    A_prev, W, B = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    # Here dA_prev => d/(dA[l-1]) (Loss function)

    return dA_prev, dW, db

# 
def linear_activation_backward(dA, cache, activation_fn):

    linear_cache = cache[0]
    activation_cache = cache[1]

    if activation_fn == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation_fn == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    # Here dZ is derivative of activation function wrt Z, i.e given A=g(Z),
    # dZ => d/dZ(Activation function)== d/dZ(g(Z))
    # dZ == dA[l] * g[l]'(Z[l])

    return dA_prev, dW, db

# 
def backward_propagation(AL, Y, caches):
    gradients = {}

    # the number of layers
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Derivative of of cross entropy loss function wrt A i.e dAL => d/dA(Loss function)== d/dA (L(A,Y))
    # => dAL = (-Y/A) + (1-Y)/(1-A)
    dAL = -(np.divide(Y, AL)) + np.divide((1 - Y), (1 - AL))

    current_cache = caches[L - 1]  # cache from the last most layer.
    dA_prev, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    gradients["dA" + str(L - 1)] = dA_prev
    gradients["dW" + str(L)] = dW_temp
    gradients["db" + str(L)] = db_temp

    # for the next last layers, the activation function is relu.
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(gradients["dA" + str(l + 1)], current_cache, "relu")
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients

# 
def update_parameters(params, gradients, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * gradients["dW" + str(l + 1)]
        parameters["B" + str(l + 1)] = parameters["B" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def run_model(X, Y, layers_dimensions, learning_rate=0.75, num_iterations=300, print_cost=True):
    costs = []
    parameters = initialize_parameters(layers_dimensions)

    # Gradient descent
    for i in range(0, num_iterations):

        # Forward propagation:
        AL, caches = forward_propagation(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        gradients = backward_propagation(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Print the cost every 100 iterations
        #if print_cost and i % 10 == 0 or i == num_iterations - 1:
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def predict(X, parameters):
    A = X
    L = len(parameters) // 2

    # loop through the hidden layers.
    for l in range(1, L):
        A_prev = A
        A, cache = hypothesis_activation(A_prev, parameters["W" + str(l)], parameters["B" + str(l)], "relu")

    # Calculate final prediction from the output layer.
    predictions, cache = linear_hypothesis(A, parameters["W" + str(L)], parameters["B" + str(L)], "sigmoid")
    predictions = np.round(predictions)

    return predictions
