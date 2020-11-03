import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def prepare_dataset():
    df = pd.read_csv('app/dataset/icml_face_data_small.csv')
    df.columns = ['label', 'usage', 'img']
    df = df.drop(columns=['usage'])

    df['img'] = df['img'].apply(lambda x: np.array(x.split(' '), dtype=int).reshape(48, 48, 1))

    x = np.array(df['img'].tolist())/255
    y = to_categorical(np.array(df['label'].tolist()))

    return x, y

def show_img(img):
    plt.figure(figsize=(6, 6))

    plt.imshow(img, cmap='gray')
    plt.axis(False)

    plt.show()

def get_emotion(label):
    emotions = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
    emotion = emotions[int(label)]

    return emotion

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

def train_model(model, epochs=10, batch_size=128, validation_split=0.2, model_name='default', es_patience=10):
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model_filepath = f'app/saves/model_{model_name}.h5'

    model_save = ModelCheckpoint(
        filepath=model_filepath, 
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=0)

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=es_patience,
        verbose=1)

    callbacks = [model_save, es]

    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)


if __name__ == '__main__':
    config_gpu()

    x, y = prepare_dataset()
    # idx = 7
    # print(get_emotion(np.argmax(y[idx])))
    # show_img(x[idx])

    model = build_model()
    # print(model.summary())

    epochs = 5
    batch_size = 256

    train_model(model, epochs=epochs, batch_size=batch_size)