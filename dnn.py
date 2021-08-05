import numpy as np
from utils.dnn_utils import sigmoid, relu, sigmoid_backward, relu_backward
from utils.adam_optimization import *
from utils.mini_batches import *
import matplotlib.pyplot as plt


def initialize_parameters(layer_dimensions):
    np.random.seed(3)
    parameters = {}

    # layer_dimensions[0] is the number of input features in a training example or --
    # the number of neurons in the training layer.

    # Total number of layers including the input layer.
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
        parameters["B" + str(l)] = np.zeros((layer_dimensions[l], 1))

        # assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
        # assert (parameters['B' + str(l)].shape == (layer_dimensions[l], 1))

    return parameters


#
def linear_forward(A, W, B):
    # Here A is the activation of the previous layer and is acting as the
    # input for the current layer. W and B are the parameters of the current layer.
    Z = np.dot(W, A) + B
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, B)
    return Z, cache


#
def linear_forward_activation(A_prev, W, B, activation_fn):
    # The linear_cache contains the A (A_prev), W , B used to calculate the Z.
    # Here the A_prev is the activation of previous layer which is acting as the input for the current layer
    # and is being used with the W and B of current layers to calculate the Z of the current layer.

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, B)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, B)
        A, activation_cache = relu(Z)

    # The activation_cache contains the value of the Z which is used to calculate the activation A
    # using the given function.

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    forward_cache = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters["W" + str(l)], parameters["B" + str(l)],
                                             activation_fn="relu")
        forward_cache.append(cache)

    AL, cache = linear_forward_activation(A, parameters["W" + str(L)], parameters["B" + str(L)],
                                          activation_fn="sigmoid")
    forward_cache.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, forward_cache


def compute_cost(AL, Y):
    m = Y.shape[1]

    # Computing the cross entropy loss
    # cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))
    #logprods = np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T)
    #cost = -1 / m * np.sum(logprods)

    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost_total = np.sum(logprobs)
    #cost = np.squeeze(cost)
    #assert (cost.shape == ())
    return cost_total


#
def linear_backward(dZ, cache):
    A_prev, W, B = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    # Here dA_prev => d/(dA[l-1]) (Loss function)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == B.shape)

    return dA_prev, dW, db


#
def linear_activation_backward(dA, cache, activation_fn):
    # linear_cache = cache[0]
    # activation_cache = cache[1]
    linear_cache, activation_cache = cache

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


def backward_propagation(AL, Y, caches):
    gradients = {}

    # print("Shape of AL is ==", AL.shape)
    # print("Shape of Y is ==", Y.shape)
    # the number of layers
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Derivative of of cross entropy loss function wrt A i.e dAL => d/dA(Loss function)== d/dA (L(A,Y))
    # => dAL = (-Y/A) + (1-Y)/(1-A)
    # dAL = -(np.divide(Y, AL)) + np.divide((1 - Y), (1 - AL))
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]  # cache from the last most layer.
    gradients["dA" + str(L - 1)], gradients["dW" + str(L)], gradients["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation_fn="sigmoid")

    # for the next last layers, the activation function is relu.
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(gradients["dA" + str(l + 1)], current_cache, "relu")
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients


#
def update_parameters(parameters, gradients, learning_rate):
    # parameters = params.copy()
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * gradients["dW" + str(l + 1)]
        parameters["B" + str(l + 1)] = parameters["B" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def run_model(X, Y, layers_dimensions, learning_rate=0.05, num_iterations=10000, print_cost=True):
    np.random.seed(1)
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
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters, costs


def predict(X, parameters):
    A = X
    L = len(parameters) // 2

    # loop through the hidden layers.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters["W" + str(l)], parameters["B" + str(l)], "relu")

    # Calculate final prediction from the output layer.
    predictions, cache = linear_forward_activation(A, parameters["W" + str(L)], parameters["B" + str(L)], "sigmoid")
    predictions = np.round(predictions)

    return predictions


def model_v2(X, Y, layers_dims, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
             beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=500, print_cost=True):
    """
        n-layer neural network model using gradient descent with Adam optimisation.

        Arguments:
        X -- input data, of shape (number of features per training example, number of examples)
        Y -- true "label" vector , of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
    seed = 1
    L = len(layers_dims)     # number of layers in the neural networks
    costs = []               # to keep track of the cost
    t = 0                    # initializing the counter required for Adam update
    m = X.shape[1]           # number of training examples

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(AL, minibatch_Y)

            # Backward propagation
            gradients = backward_propagation(AL, minibatch_Y, caches)

            # Update parameters
            t = t + 1                # Adam counter
            parameters, v, s, _, _ = update_parameters_with_adam(parameters, gradients, v, s,
                                                                 t, learning_rate, beta1, beta2, epsilon)

        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 10 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 10 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


