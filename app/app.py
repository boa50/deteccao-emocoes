import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
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

def show_img(img, rgb=False, title=''):
    if rgb:
        img = img[:, :, ::-1]
        cmap = None
    else:
        cmap = 'gray'

    plt.figure(figsize=(6, 6))

    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis(False)

    plt.show()

def get_emotions():
    return ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

def get_emotion(label):
    emotions = get_emotions()
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

def train_model(x_train, x_val, y_train, y_val, model, epochs=10, batch_size=128, model_name='default', es_patience=10):
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

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks).history

    return history

def plot_curvas(dados, labels, title, ylabel):
    fig = plt.figure(figsize=[16,9])
    ax = fig.add_subplot(111)

    ax.plot(range(1, len(dados[0]) + 1), dados[0], c='r', label=labels[0])
    ax.plot(range(1, len(dados[1]) + 1), dados[1], c='b', label=labels[1])

    ax.set_title(title, fontdict={'fontsize': 20})
    ax.set_xlabel('Épocas', fontdict={'fontsize': 15})
    ax.set_ylabel(ylabel, fontdict={'fontsize': 15})
    ax.legend(fontsize=12)
    plt.show()

def plot_confusion_matrix(x_val, y_val, model):
    y_pred = model.predict(x_val)
    conf_matrix = confusion_matrix([np.argmax(x) for x in y_val], [np.argmax(x) for x in y_pred])
    labels = get_emotions()

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    ax.set_title("Matriz de Confusão", fontsize=20)
    ax.set_ylabel('Classe Verdadeira', fontsize=15)
    ax.set_xlabel('Classe Predita', fontsize=15)
    plt.show()

def plot_analises(history, x_val, y_val, model):
    dados = [history['loss'], history['val_loss']]
    labels = ['Perda de treino', 'Perda de validação']
    title = 'Treinamento - Perdas'
    ylabel = 'Perda'
    plot_curvas(dados, labels, title, ylabel)

    dados = [history['accuracy'], history['val_accuracy']]
    labels = ['Acurácia de treino', 'Acurácia de validação']
    title = 'Treinamento - Acurácias'
    ylabel = 'Acurácia'
    plot_curvas(dados, labels, title, ylabel)

    plot_confusion_matrix(x_val, y_val, model)

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_coords = face_cascade.detectMultiScale(img, minNeighbors=8)
    
    face_imgs = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for face in faces_coords:
        (x, y, w, h) = face
        face_imgs.append(img_gray[y+1:y+h, x+1:x+w])

    return faces_coords, face_imgs

def prepare_face(face):
    network_input_img_size = (48, 48)

    if face.shape[0] > network_input_img_size[0]:
        face = cv2.resize(face, network_input_img_size, interpolation=cv2.INTER_AREA)
    else:
        face = cv2.resize(face, network_input_img_size, interpolation=cv2.INTER_CUBIC)

    face = np.expand_dims(face, -1)

    return face

def prepare_img(img_path):
    img = cv2.imread(img_path)
    faces_coords, face_imgs = detect_faces(img)

    faces = []
    for face in face_imgs:
        faces.append(prepare_face(face))

    return img, faces_coords, np.array(faces)

def show_emotions(img, faces_coords, predicts):
    img_detected = img.copy()

    font_scale = max(img.shape[1] // 600, 1)

    for coords, predict in zip(faces_coords, predicts):
        emocao = get_emotion(np.argmax(predict))

        (x, y, w, h) = coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_pos = x - 1*font_scale
        y_pos = y - 3*font_scale
        font_size = 0.5*font_scale
        font_thick = int(1*np.ceil(font_scale / 2))

        cv2.rectangle(img_detected, (x, y), (x+w, y+h), (0, 255, 0), font_thick)
        cv2.putText(img_detected, emocao, (x_pos, y_pos), font, font_size, (0, 255, 0), font_thick, cv2.LINE_AA)

    show_img(img_detected, rgb=True)


if __name__ == '__main__':
    config_gpu()

    # x, y = prepare_dataset()
    # idx = 7
    # print(get_emotion(np.argmax(y[idx])))
    # show_img(x[idx])
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=50)

    # model = build_model()
    # print(model.summary())

    # epochs = 5
    # batch_size = 256
    # history = train_model(x_train, x_val, y_train, y_val, model, epochs=epochs, batch_size=batch_size)

    # plot_analises(history, x_val, y_val, model)

    model = load_model('app/saves/model_default.h5')
    # model.predict(x_val)

    img_path = 'app/dataset/imgs/01.jpeg'
    img, faces_coords, faces = prepare_img(img_path)

    predicts = model.predict(faces)
    show_emotions(img, faces_coords, predicts)