from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from keras.models import Model

# Model Creater Helper Func
# Helper For 1st half
def goingDown(inputLayer, filters, pool=True):
    conv1 = Conv2D(filters, 3, activation='relu', padding="same")(inputLayer)
    conv2 = Conv2D(filters, 3, activation='relu', padding="same")(conv1)
    if pool:
        pool1 = MaxPooling2D(2)(conv2)
        return conv2, pool1
    return conv2

# Helper For 1st half
def goingUp(inputLayer, mergeLayer, filters):
    upSample1 = UpSampling2D()(inputLayer)
    upConv1 = Conv2D(filters, 2, padding="same")(upSample1)
    concat1 = Concatenate(axis = 3)([mergeLayer, upConv1])
    conv1 = Conv2D(filters, 3, activation='relu', padding="same")(concat1)
    conv2 = Conv2D(filters, 3, activation='relu', padding="same")(conv1)
    return conv2

# Helper For Output Layer
def goingOut(inputLayer):
    conv1 = Conv2D(1, 1, activation='sigmoid')(inputLayer)
    return conv1

def createModel():
    filters = 64
    inputLayer = Input(shape = [512, 512, 3])
    mergeLayer = []
    
    # Total Down 4 | Bottom 1 |Up 4
    
    # Down 1
    mergeLayer1, down1 = goingDown(inputLayer, filters)
    filters *= 2
    mergeLayer.append(mergeLayer1)
    
    # Down 2
    mergeLayer2, down2 = goingDown(down1, filters)
    filters *= 2
    mergeLayer.append(mergeLayer2)
    
    # Down 3
    mergeLayer3, down3 = goingDown(down2, filters)
    filters *= 2
    mergeLayer.append(mergeLayer3)
    
    # Down 4
    mergeLayer4, down4 = goingDown(down3, filters)
    filters *= 2
    mergeLayer.append(mergeLayer4)
    
    # Bottom
    bottom = goingDown(down4, filters, pool = False)    
    
    # Up 1
    filters = filters / 2
    up1 = goingUp(bottom, mergeLayer[-1], filters=int(filters))
    
    # Up 2
    filters = filters / 2
    up2 = goingUp(up1, mergeLayer[-2], filters=int(filters))
    
    # Up 3
    filters = filters / 2
    up3 = goingUp(up2, mergeLayer[-3], filters=int(filters))
    
    # Up 4
    filters = filters / 2
    up4 = goingUp(up3, mergeLayer[-4], filters=int(filters))
    
    # Output
    out = goingOut(up4)
    
    # Create Model
    model = Model(inputLayer, out)
    
    return model