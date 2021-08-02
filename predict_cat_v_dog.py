import glob
from PIL import Image
import numpy as np
import dnn

def load_images(path):
    images = []
    for img in glob.glob(path):
        img = Image.open(img)
        img = img.resize((64, 64))
        img_arr = np.array(img)
        images.append(img_arr)

    m = len(images)
    images_dataset = np.array(images).reshape(m, -1).T
    images_dataset = images_dataset / 255

    return images, images_dataset


parameters = np.load("cat_v_dog_dnn_learned_parameters_v1.1.npy", allow_pickle=True).flat[0]
images, images_dataset = load_images("datasets/test_images/*.JPG")
predictions = dnn.predict(images_dataset, parameters)
print(predictions)

for i in range(len(images)):
    img = Image.fromarray(images[i], 'RGB')
    if predictions[0][i] == 1:
        text = "This is a cat image !!!"
    if predictions[0][i] ==0:
        text = "This is a dog image !!!"
    img.show(title=text)

