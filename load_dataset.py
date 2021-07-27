import numpy as np
import glob
from PIL import Image
from random import shuffle
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def load_images():
    dataset = []
    i = 0
    j = 0
    for img in glob.glob(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning"
                         r"\Deep Neural network\datasets\cats_mini\*.JPG"):
        i += 1
        img = Image.open(img)
        img_arr = np.array(img)
        dataset.append([img_arr, 1])
        if i%100 == 0:
            print(i, " cat images loaded successfully !!")

    for img in glob.glob(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning"
                         r"\Deep Neural network\datasets\dogs_mini\*.JPG"):
        j += 1
        img = Image.open(img)
        img_arr = np.array(img)
        dataset.append([img_arr, 0])
        if j%100 == 0:
            print(j, " dog images loaded successfully !!")
    shuffle(dataset)
    shuffle(dataset)
    shuffle(dataset)
    shuffle(dataset)
    shuffle(dataset)
    return dataset

def load_train_test_sets():
    dataset = load_images()
    X_train = []
    Y_train = []
    for i in range(1000):
        X_train.append(dataset[i][0])
        Y_train.append(dataset[i][1])

    X_test = []
    Y_test = []
    for j in range(200):
        X_test.append(dataset[1000+j][0])
        Y_test.append(dataset[1000+j][1])

    X_train = np.array(X_train).reshape(1000, -1).T
    Y_train = np.array(Y_train).reshape(1000, 1).T

    X_test = np.array(X_test).reshape(200, -1).T
    Y_test = np.array(Y_test).reshape(200, 1).T

    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, Y_train, X_test, Y_test


#np.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning\Deep Neural network\datasets\X_train_set_data", X_train)
#np.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning\Deep Neural network\datasets\Y_train_set_data", Y_train)
#np.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning\Deep Neural network\datasets\X_test_set_data", X_test)
#np.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning\Deep Neural network\datasets\Y_test_set_data", Y_test)
"""
data = load_images()
for i in range(10):
    img = Image.fromarray(data[i][0], 'RGB')

    if data[i][1] == 1:
        text= "This is a CAT !!"
        print(text)
    if data[i][1] ==0:
        text= "this is a DOG !!"
        print(text)
    img.show(title=text)
"""