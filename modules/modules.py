import cv2
import os
from flask import json
import numpy as np

def hello():
    return "Hello, World!"

def content():
    return '''
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
            '''

def capture_faces_from_image(person_name, team, career, img_data_list, data_path='data'):
    person_path = os.path.join(data_path, person_name)
    
    if not os.path.exists(person_path):
        os.makedirs(person_path)
    
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        with open('instance/players.json', 'r') as f:
            players = json.load(f)
    except FileNotFoundError:
        players = {}

    if person_name not in players:
        players[person_name] = {
            'path': person_path,
            'team': team,
            'career': career,
            'images': []
        }

    count = len(players[person_name]['images'])
    
    for img_data in img_data_list:
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        faces = face_classif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = img_data[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            image_name = f'rostro_{count}.jpg'
            cv2.imwrite(os.path.join(person_path, image_name), rostro)
            players[person_name]['images'].append(image_name)
            count += 1

    with open('instance/players.json', 'w') as f:
        json.dump(players, f, indent=4)

def recognize_faces_from_image(img, data_path='data'):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classif.detectMultiScale(gray, 1.3, 5)
    result = "Face not recognized"

    with open('instance/players.json', 'r') as f:
        players = json.load(f)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        label, confidence = face_recognizer.predict(rostro)
        if confidence < 100:
            for name, data in players.items():
                if int(label) in [int(i.split('_')[1].split('.')[0]) for i in data['images']]:
                    result = f"Bienvenido Jugador {name} de {data['team']}, de la {data['career']}"
                    break
        else:
            result = "Face not recognized"
    
    return result