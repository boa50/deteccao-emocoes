import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

import utils
import detection_cnn as detection
import standard as std
import pcn

if __name__ == '__main__':
    utils.config_gpu()

    ### Treinamento do modelo
    # x, y = utils.prepare_dataset()
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=50)

    # model = detection.build_model()
    # epochs = 1000
    # batch_size = 128
    # history = detection.train_model(x_train, x_val, y_train, y_val, model, epochs=epochs, batch_size=batch_size)

    # detection.plot_analises(history, x_val, y_val, model)


    ### Utilização de abordagens padrão
    # model = load_model('app/saves/model_default_58526.h5')
    # detection.plot_analises([], x_val, y_val, model)
    # print(model.summary())

    # img_path = 'app/dataset/imgs/multi_rotated.jpg'
    # img, faces_coords, faces = std.prepare_img(img_path, detection='opencv')
    # img, faces_coords, faces = std.prepare_img(img_path, detection='mtcnn')

    # predicts = model.predict(faces)
    # std.show_emotions(img, faces_coords, predicts)
    

    ### Utilização da PCN
    # img = cv2.imread(img_path)
    # detector = pcn.get_detector()
    # windows = detector.DetectAndTrack(img)

    # for win in windows:
    #     croped = pcn.prepare_img(img, win)
    #     predict = model.predict(np.array([croped]))
    #     pcn.DrawFace(win, img, predict=predict)
    
    # utils.show_img(img, rgb=True)