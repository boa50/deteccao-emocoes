import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

import utils

def detect_faces_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_coords = face_cascade.detectMultiScale(img, minNeighbors=8)
    
    face_imgs = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for face in faces_coords:
        (x, y, w, h) = face
        face_imgs.append(img_gray[y+1:y+h, x+1:x+w])

    return faces_coords, face_imgs

def detect_faces_mtcnn(img):
    faces_coords = []
    face_imgs = []

    detector = MTCNN()
    faces = detector.detect_faces(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for face in faces:
        x, y, w, h = face['box']
        faces_coords.append((x, y, w, h))
        face_imgs.append(img_gray[y+1:y+h, x+1:x+w])

    return faces_coords, face_imgs

# def prepare_face(face):
#     network_input_img_size = (48, 48)

#     if face.shape[0] > network_input_img_size[0]:
#         face = cv2.resize(face, network_input_img_size, interpolation=cv2.INTER_AREA)
#     else:
#         face = cv2.resize(face, network_input_img_size, interpolation=cv2.INTER_CUBIC)

#     face = np.expand_dims(face, -1)

#     return face

def prepare_img(img_path, detection='opencv'):
    img = cv2.imread(img_path)

    if detection == 'opencv':
        faces_coords, face_imgs = detect_faces_opencv(img)
    elif detection == 'mtcnn':
        faces_coords, face_imgs = detect_faces_mtcnn(img)
    else:
        print('Método de detecção de faces não disponível')
        return None, None, None

    faces = []
    for face in face_imgs:
        faces.append(utils.prepare_face(face))

    return img, faces_coords, np.array(faces)

def show_emotions(img, faces_coords, predicts):
    img_detected = img.copy()

    font_scale = max(img.shape[1] // 600, 1)
    square_color = (50, 205, 50)
    text_color = (255, 255, 255)

    for coords, predict in zip(faces_coords, predicts):
        emocao = utils.get_emotion(np.argmax(predict))

        (x, y, w, h) = coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_pos = x + 1*font_scale
        y_pos = y - 3*font_scale
        font_size = 0.5*font_scale
        font_thick = int(1*np.ceil(font_scale / 2))

        cv2.rectangle(img_detected, (x, y), (x+w, y+h), square_color, font_thick)
        cv2.rectangle(img_detected, (x, y - (16*font_scale)), (x+w, y), square_color, -1)
        cv2.putText(img_detected, emocao, (x_pos, y_pos), font, font_size, text_color, font_thick, cv2.LINE_AA)

    utils.show_img(img_detected, rgb=True)