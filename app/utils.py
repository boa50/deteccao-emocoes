import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def prepare_dataset():
    df = pd.read_csv('app/dataset/icml_face_data.csv')
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

### Implementação baseada na contida no artigo https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))