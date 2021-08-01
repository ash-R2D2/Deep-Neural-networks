from dnn import forward_propagation, backward_propagation, initialize_parameters, compute_cost
from grad_check_utils import *
import numpy as np

from PIL import Image


def gradient_check(parameters, gradients, X, Y,layer_dims, epsilon=1e-7):
    parameters_values, _ = dictionary_to_vector(parameters)



    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    print(" number of parameters = ", num_parameters)

    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    print(" J+ and J- initialization successfull")

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus[i], _ = forward_propagation(X, vector_to_dictionary(theta_plus, layer_dims))


        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        theta_minus = np.copy(parameters_values)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus[i], _ = forward_propagation(X, vector_to_dictionary(theta_minus, layer_dims))

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        if i%100 ==0 :
            print(" J+- for parameter number ", i, " Calculated.......")

    print("Final gradapprox calculated...... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.......")
    print("Grad approx is =        ", gradapprox)
    print("backward prop grad is = ", grad)
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    return difference

n_x = int(24 * 24 * 3)
layer_dims = [n_x, 200, 70, 40, 1]

img = Image.open(r"C:\Users\avina\Documents\GitHub\Deep-Neural-networks\datasets\single_cat_image\cat_00012.JPG")
img_cropped = img.resize((24,24))
img_cropped.show()
img_arr = np.array(img_cropped)

label = [1]
X = img_arr.reshape(n_x,1)
Y = np.array(label).reshape(1,1)

print(X.shape)
print(Y.shape)


parameters = initialize_parameters(layer_dims)
print("Parameters initializaion successfull..")
AL, cache = forward_propagation(X, parameters)
print("Forward propagation Successfull....")
cost = compute_cost(AL, Y)
print("Compute cost Successfull....")
gradients = backward_propagation(AL, Y, cache)
print("Backward propagation Successfull....")
difference = gradient_check(parameters, gradients, X, Y, layer_dims,  1e-7)

print("The difference is = ", difference)