from model import createModel
from imagePreProcessing import dataGenerator

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time

def trainUnetModel(batchSize = 5):
    #NAME = f'Unet-{int(time.time())}'
    #tensorboard = TensorBoard(log_dir = f'logs/{NAME}')

    model = createModel()
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    trainGenerator = dataGenerator(batchSize)
    #valGenerator = dataGenerator(batchSize, val = True)
    model.fit_generator(trainGenerator, steps_per_epoch = 30, epochs = 10)#, validation_data = valGenerator, validation_steps = 10, callbacks = [tensorboard])

    return model
    

model = trainUnetModel(1)
model.save("modelSaved/UNet.h5")