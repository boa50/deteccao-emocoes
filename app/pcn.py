import numpy as np
import cv2
from PyPCN import PCN

import utils

### Implementação baseada na do github do FaceKit
def find_points(win):
    x1 = win.x
    y1 = win.y
    x2 = win.width + win.x - 1
    y2 = win.width + win.y - 1
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2
    angle = win.angle
    R = cv2.getRotationMatrix2D((centerX,centerY),angle,1)
    pts = np.array([[x1,y1,1],[x1,y2,1],[x2,y2,1],[x2,y1,1]], np.int32)
    pts = (pts @ R.T).astype(int)
    pts = pts.reshape((-1,1,2))

    return pts, angle, x1, y1

def get_detector():
    root_path = "repositorio_externo/FaceKit/PCN/"
    detection_model_path = root_path + "model/PCN.caffemodel"
    pcn1_proto = root_path + "model/PCN-1.prototxt"
    pcn2_proto = root_path + "model/PCN-2.prototxt"
    pcn3_proto = root_path + "model/PCN-3.prototxt"
    tracking_model_path = root_path + "model/PCN-Tracking.caffemodel"
    tracking_proto = root_path + "model/PCN-Tracking.prototxt"
    embed_model_path = root_path + "model/resnetInception-128.caffemodel"
    embed_proto = root_path + "model/resnetInception-128.prototxt"

    detector = PCN(detection_model_path, pcn1_proto, pcn2_proto, pcn3_proto,
			tracking_model_path, tracking_proto, embed_model_path, embed_proto,
			15,1.45,0.5,0.5,0.98,30,0.9,0)

    return detector

def prepare_img(img, win):
    pts, angle, _, _ = find_points(win)

    x,y,w,h = cv2.boundingRect(pts)
    croped = img[y:y+h, x:x+w].copy()

    pts_mask = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts_mask], -1, (255, 255, 255), -1, cv2.LINE_AA)

    croped = cv2.bitwise_and(croped, croped, mask=mask)

    croped = utils.rotate_bound(croped, angle)

    gray = cv2.cvtColor(croped,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)

    croped = gray[y+1:y+h-1,x+1:x+w-1]
    croped = utils.prepare_face(croped)

    return croped

### Implementação baseada na do github do FaceKit
def DrawFace(win, img, predict=None):
    if predict is None:
        emocao = ''
    else:
        emocao = utils.get_emotion(np.argmax(predict))

    square_color = (50, 205, 50)

    pts, _, x1, y1 = find_points(win)
    cv2.polylines(img, [pts], True, square_color, thickness=5)
    cv2.putText(img, emocao, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 15, cv2.LINE_AA)
    cv2.putText(img, emocao, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 10, cv2.LINE_AA)