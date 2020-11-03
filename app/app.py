import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def prepare_dataset():
    df = pd.read_csv('app/dataset/icml_face_data_small.csv')
    df.columns = ['label', 'usage', 'img']
    df = df.drop(columns=['usage'])

    emotions = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

    df['emotion'] = df['label'].apply(lambda x: emotions[int(x)])
    df['img'] = df['img'].apply(lambda x: np.array(x.split(' '), dtype=int).reshape(48, 48, 1))

    return df

def show_img(img):
    plt.figure(figsize=(6, 6))

    plt.imshow(img, cmap='gray')
    plt.axis(False)

    plt.show()

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
    # config_gpu()

    df = prepare_dataset()

    print(df.head())

    show_img(df['img'][5])
    # model = build_model()