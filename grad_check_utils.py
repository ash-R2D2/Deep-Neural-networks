import numpy as np

def dictionary_to_vector(parameters):
    keys = []
    count = 0
    for key in ["W1", "B1", "W2", "B2", "W3", "B3", "W4", "B4"]:
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta, layer_dims):
    parameters = {}

    c = layer_dims[1]*layer_dims[0]
    parameters["W1"] = theta[:c].reshape((layer_dims[1],layer_dims[0]))
    parameters["B1"] = theta[c: c+ layer_dims[1]].reshape((layer_dims[1], 1))

    d = c + layer_dims[1] + layer_dims[1]*layer_dims[2]
    parameters["W2"] = theta[c + layer_dims[1]:d].reshape((layer_dims[2], layer_dims[1]))
    parameters["B2"] = theta[d: d+layer_dims[2]].reshape((layer_dims[2], 1))

    e = d+layer_dims[2] + layer_dims[2]*layer_dims[3]
    parameters["W3"] = theta[d+layer_dims[2]:e].reshape((layer_dims[3], layer_dims[2]))
    parameters["B3"] = theta[e: e+layer_dims[3]].reshape((layer_dims[3], 1))

    f = e+layer_dims[3] + layer_dims[3]*layer_dims[4]
    parameters["W4"] = theta[e+layer_dims[3]:f].reshape((layer_dims[4], layer_dims[3]))
    parameters["B4"] = theta[f: f + layer_dims[4]].reshape((layer_dims[4], 1))

    return parameters

def gradients_to_vector(gradients):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3", "dW4", "db4"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

