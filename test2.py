import cv2,os
from PIL import Image
from fastai.vision import *
import warnings
warnings.filterwarnings("ignore")
learn = load_learner('D:\\data\\face\\images', 'export.pkl')

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    cv2.imwrite('D:\\data\\face\\images\\test\\frame.jpg',rgb_frame)
    img_path='D:\\data\\face\\images\\test\\frame.jpg'

    c=learn.predict(open_image(img_path))[0]
    if int(c)==0:
        s='Intruder Detected... Call 911'
    else:s='Not Intruder'

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, s, (50 + 6, 50 - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
