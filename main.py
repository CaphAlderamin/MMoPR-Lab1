import face_recognition as fr
import numpy as np
import cv2
import os

faces_path = "E:\\Study\ТПУ МАГИСТРАТУРА\\2 Семестр\\Математические методы распознавания образов\\Лаб 1\\Faces"


def get_face_encodings():
    face_names = os.listdir(f"{faces_path}\\known")
    face_encodings = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}\\known\\{name}")
        face_encodings.append(fr.face_encodings(face)[0])

        face_names[i] = name.split(".")[0]

    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

video = cv2.VideoCapture(0)

scl = 2

while True:
    success, image = video.read()

    resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)