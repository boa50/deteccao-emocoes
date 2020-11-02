import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def build_model():
    img_shape = (48, 48, 1)

    img_input = Input(shape=img_shape)
    x = Conv2D(32, 3, padding='same', activation='relu')(img_input)
    x = MaxPool2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    out = Dense(7, activation='softmax')(x)

    model = Model(img_input, out) 

    return model

if __name__ == '__main__':
    config_gpu()

    model = build_model()
    print(model.summary())
