import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG19

import utils

def build_model():
    img_shape = (48, 48, 1)

    img_input = Input(shape=img_shape)
    x = Conv2D(64, 3, padding='same', activation='relu')(img_input)
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(1024)(x)
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
        monitor='val_accuracy',
        mode='max',
        verbose=1)

    es = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
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

def plot_confusion_matrix(y_val, y_pred):
    conf_matrix = confusion_matrix(y_val, y_pred)
    labels = utils.get_emotions()

    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    ax.set_title("Matriz de Confusão", fontsize=20)
    ax.set_ylabel('Classe Verdadeira', fontsize=15)
    ax.set_xlabel('Classe Predita', fontsize=15)
    plt.show()

def plot_analises(history, x_val, y_val, model):
    if len(history) > 0:
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

    y_pred = model.predict(x_val)
    y_val = [np.argmax(x) for x in y_val]
    y_pred = [np.argmax(x) for x in y_pred]
    labels = utils.get_emotions()
    
    plot_confusion_matrix(y_val, y_pred)
    print(classification_report(y_val, y_pred, target_names=labels))