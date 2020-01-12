import cv2
import numpy as np
import os


def detected():
    names = dict()
    path = '/home/nastya/PycharmProjects/tello/dataset'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for image in imagePaths:
        id_user = int(os.path.split(image)[-1].split(".")[1])
        name = str(os.path.split(image)[-1].split(".")[0])
        if name not in names:
            names[id_user] = name
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('/home/nastya/PycharmProjects/tello/trainer.yml')
    faceCascade = cv2.CascadeClassifier("/home/nastya/PycharmProjects/tello/haarcascades/haarcascade_frontalface_default.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = cv2.VideoCapture(-1)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
           )

        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Detected user', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detected()