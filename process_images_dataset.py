import numpy as np
from PIL import Image
import glob


def square_crop_Image(img):
    w, h = img.size
    left = 0
    top = 0
    bottom = 0
    right = 0
    if w > h:
        diff = int((w-h)/2)
        left = diff
        right = int(w - diff)
        bottom = h
    else:
        diff = int(h - w)/2
        right = w
        top = diff
        bottom = int(h-diff)

    sq_img = img.crop((left, top, right, bottom))
    return sq_img


images = []
ext = ['JPG']    # Add image formats here
imdir = r"C:\Users\avina\Documents\Deep Learning\datasets\cats-dogs\dataset\training_set\dogs"
files = []
[files.extend(glob.glob(imdir + '\*.' + e)) for e in ext]
j=0
for img in files:
    j+=1
    img = Image.open(img)
    img = img.convert('RGB')
    img = square_crop_Image(img)
    img = img.resize((256,256))
    img_arr = np.array(img)
    if len(img_arr.shape) != 3:
        print("Found the outlier ", img_arr.shape)
        continue
    images.append(img_arr)

    if j%500 == 0:
        print("Image no. ",j," loaded")


for i in range(len(images)):
    if len(images[i].shape) != 3:
        print("Found the outlier ", images[i].shape)
        continue
    data = Image.fromarray(images[i])
    img_name = str(i).rjust(5, "0") + ".JPG"
    if i%500 == 0:
        print("image no. ", i ," saved")
    data.save(r"C:\Users\avina\Documents\Deep Learning\Neural networks and deep learning"
    r"\Deep Neural network\datasets\dogs_256px\dog_"+img_name)

