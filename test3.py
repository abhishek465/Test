import cv2,os
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def predict(model, test_image_name):
    transform = data_transforms['test']
    test_image = Image.open(test_image_name)
    test_image_tensor = transform(test_image)
     
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        #print("Output class :  ", topclass.numpy()[0][0])
        return topclass.numpy()[0][0]
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

#learn = load_learner('D:\\data\\face\\images', 'export.pkl')
model = load_checkpoint('D:\\data\\face\\images\\checkpoint.pth')
#t=predict(learn,'D:\\data\\face\\images\\WIN_20190828_16_57_59_Pro.jpg')
#print(t)

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    cv2.imwrite('D:\\data\\face\\images\\test\\frame.jpg',rgb_frame)
    img_path='D:\\data\\face\\images\\test\\frame.jpg'

    #c=learn.predict(open_image(img_path))[0]
    c=predict(model,img_path)
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
