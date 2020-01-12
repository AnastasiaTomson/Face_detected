import cv2
import numpy as np
from PIL import Image
import os
from face_3 import detected


def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier("/home/nastya/PycharmProjects/tello/haarcascades/haarcascade_frontalface_default.xml")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


def training():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n Идет обучение, подождите пару секунд ...")
    faces, ids = getImagesAndLabels('/home/nastya/PycharmProjects/tello/dataset')
    recognizer.train(faces, np.array(ids))

    recognizer.write('/home/nastya/PycharmProjects/tello/trainer.yml')

    print("\n Лицо записано в базу")

    if input() == '':
        detected()
    else:
        exit()

