import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascades_frontalface_default.xml')

def detect_faces(image_path, output_dir):
    # Load the image using matplotlib (RGB format)
    image = plt.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use the classifier to detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Save the headshots of the detected faces
    i = 1
    for (x,y,w,h) in faces:
        headshot = image[y:y+h, x:x+w]
        #headshot = cv2.cvtColor(headshot, cv2.COLOR_BGR2RGB)
        headshot = Image.fromarray(headshot)
        headshot.save(os.path.join(output_dir, 'face_{}.jpg'.format(i)))
        i += 1

    return len(faces)


image_path = "C:/Users/astro/Documents/GitHub/Data-engineer-wavebreak/Images/diversity people 2.jpeg"
