import os
import numpy as np

from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
import matplotlib.pyplot as plt

from keras.models import load_model

# Load Model
model = load_model('modelSaved/UNet.h5')

# Create Predict Data & predict & save
predImages = os.listdir('topredict')

for img in predImages:
    imgs = []
    originalImage = load_img(f'topredict/{img}')
    resizedImage = imresize(originalImage, [512, 512, 3])
    arrayImage = img_to_array(resizedImage) / 255
    imgs.append(arrayImage)

    arrayImageNP = np.array(imgs)

    predArr = model.predict(arrayImageNP)

    imgName = img.split('.')[0]
    plt.imsave(f'predicted/{imgName}_mask.png', predArr[0, :, :, 0], cmap = 'gray')

    predImg = np.where(predArr > .5, 255, 0)
    plt.imsave(f'predicted/{imgName}_mask_5.png', predImg[0, :, :, 0], cmap = 'gray')

    predImg = np.where(predArr > .75, 255, 0)
    plt.imsave(f'predicted/{imgName}_mask_75.png', predImg[0, :, :, 0], cmap = 'gray')