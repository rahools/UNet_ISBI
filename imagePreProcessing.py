import os
import numpy as np
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
from sklearn.model_selection import train_test_split


# Seperate Multi-Image TIF
def tifSep():
    tifs = os.listdir('input')

    for tif in tifs:
        dirName = tif.split('.')[0]
        try:
            os.makedirs(f'input/{dirName}')
        except:
            pass
        img = Image.open(f'input/{tif}')
        for i in range(30):
            img.seek(i)
            try:
                img.save(f'input/{dirName}/img{i}.png')
            except:
                pass

# Data Generator for Training and Validation data
volumeImages = os.listdir(f'input/train-volume/')
trainImages, validationImages = train_test_split(volumeImages, train_size=0.8, test_size=0.2)

def dataGenerator(batch_size = 5, dims = [512, 512], val = False):
    if val:
        imageSet = volumeImages
    else:
        imageSet = validationImages

    while True:
        ix = np.random.choice(np.arange(len(imageSet)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            originalImage = load_img(f'input/train-volume/{imageSet[i]}')
            resizedImage = imresize(originalImage, dims + [3])
            arrayImage = img_to_array(resizedImage) / 255
            imgs.append(arrayImage)
            
            # masks
            originalMask = load_img(f'input/train-labels/{imageSet[i]}')
            resizedMask = imresize(originalMask, dims + [3])
            arrayMask = img_to_array(resizedMask) / 255
            labels.append(arrayMask[:, :, 0])
            
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels.reshape(-1, dims[0], dims[1], 1)