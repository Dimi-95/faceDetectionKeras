from ctypes import resize
import cv2 as cv 
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

video = cv.VideoCapture(0)
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("C:\\Users\\DimitriosKasderidis\\Desktop\\keras\\test.jpg")
show_result = False
size = (224, 224)

font = cv.FONT_HERSHEY_SIMPLEX


#Wenn man mehrere Personen finden will, einfach ein neues Model trainieren und mit der gleichen Abfrage aufrufen

while(True):
    ret, image_video = video.read()
    image = Image.open("C:\\Users\\DimitriosKasderidis\\Desktop\\keras\\test.jpg")
    
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    video_array = np.asarray(image)
    normalized_video_array = (video_array.astype(np.float32) / 127.0) - 1
    data[0] =  normalized_video_array
    prediction = model.predict(data)
    yes = prediction[0][0]


    if yes >= 0.8:
        show_result = True
    else:
        show_result = False

    if show_result == False:
        cv.putText(image_video, 'NICHT DIMI', (50, 50), font, 2, (0, 0, 255), 2, cv.LINE_4)
    if show_result == True:
        cv.putText(image_video, 'DIMI', (50, 50), font, 2, (0, 255, 0), 2, cv.LINE_4)


    cv.imshow("Detection", image_video)
    cv.imwrite("C:\\Users\\DimitriosKasderidis\\Desktop\\keras\\test.jpg", image_video)
    if cv.waitKey(1) &0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
