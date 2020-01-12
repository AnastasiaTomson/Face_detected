import cv2
from face_2 import training

cam = cv2.VideoCapture(-1)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('/home/nastya/PycharmProjects/tello/haarcascades/haarcascade_frontalface_default.xml')

face_id = input('\n Введите id пользователя: ')
print('\n Введите имя')
name = str(input())
print("\n Иницилизация лица. Посмотрите на камеру и подождите ...")
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        count += 1

        cv2.imwrite("/home/nastya/PycharmProjects/tello/dataset/{}.{}.{}.jpg"
                    .format(str(name), str(face_id), str(count)),
                    gray[y:y+h, x:x+w])
        cv2.imshow('Scan face', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break


cam.release()
cv2.destroyAllWindows()
print(training())