from tkinter import *
from tkinter import ttk
import os,random,subprocess
from cv2 import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


root = Tk()
root.attributes("-fullscreen",True)

#root.geometry('1500x1500')
root.title("Emotion Melody")

text = Label(root,text = "Let's make some noise!")
text.pack()
photo = PhotoImage(file='icon.gif')
labelPhoto = Label(root,image=photo)
labelPhoto.pack()

def scan():
    # load model
    model = model_from_json(open("/fer.json", "r").read())
    # load weights
    model.load_weights('/fer.h5')

    face_haar_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (600, 350))
        cv2.imshow('Emotion Melody', resized_img)
        key = cv2.waitKey(30) & 0xff
        if key == 13:  # wait until 'enter' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows

    mp = '/Applications/VLC.app/Contents/MacOS/VLC'
    if predicted_emotion == "happy":
        file = '/Users/ashmika/Desktop/face-detection/songs/happy/'
        lab = Label(root, text="You looked happy, so an energetic happy playlist was created!")
        lab.pack()
        subprocess.call([mp, file])

    if predicted_emotion == "sad":
        lab=Label(root,text = "You looked sad, a sad songs playlist was created")
        lab.pack()
        file = '/Users/ashmika/Desktop/face-detection/songs/sad/'
        subprocess.call([mp, file])

    if predicted_emotion == "angry":
        lab=Label(root,text = "You looked angry, hope that playlist put you in a good mood!")
        lab.pack()
        file = '/Users/ashmika/Desktop/face-detection/songs/neutral/'
        subprocess.call([mp, file])

    if predicted_emotion == "neutral":
        lab=Label(root,text = "You were calm, hope that playlist matched your vibe!")
        lab.pack()
        file = '/Users/ashmika/Desktop/face-detection/songs/neutral/'
        subprocess.call([mp, file])



btn = ttk.Button(root,text="Scan face",command = scan)
btn.pack()
label=Label(root,text = "Press the Enter Key when you feel your emotion has been detected and a playlist based on your emotion will be created in your music player!!\nClose the music player and scan window and go again!\n\n\n")
label.pack()

root.mainloop()
