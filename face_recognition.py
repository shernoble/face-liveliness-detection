import cv2
import numpy as np
import os 
import random

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
facecascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(facecascadePath);
smileCascadePath="Cascades/haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smileCascadePath);
eyeCascadePath="Cascades/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(eyeCascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Kavya','Satansu'] 

blink_count=0;
n = random.randint(1, 10)

# Initialize and start realtime video capture
first_read=True

live_smile=False
live_blink=False

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()

    text = "Blink "+str(n)+" times to detect"
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.5  # Reduced font size
    font_thickness = 1
    text_color = (0, 0, 255)  # Red color

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    frame_height, frame_width = img.shape[:2]

    text_x = 10
    text_y = frame_height - 10 

    # Put text on the video frame


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # Get the region of interest (ROI) for face in grayscale and color
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect smiles within the face ROI
        smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            live_smile=True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Smile', (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(50, 50))
        for (ex,ey,ew,eh) in eyes:
                
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255, 153, 255),2)
                if len(eyes) >= 2:
                    if blink_count<n:
                        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                    if first_read:
                        cv2.putText(img, "Eye's detected, press s to check blink", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                            1, (255, 0, 0), 2)
                    else:
                        
                        cv2.putText(img, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                            1, (255, 255, 255), 2)
                else:
                    if first_read:
                        cv2.putText(img, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                            1, (255, 0, 255), 2)
                    else:
                        live_blink=True
                        blink_count+=1
                        print("blink count = ",blink_count)
                        cv2.putText(img, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                            1, (0, 0, 0), 2)
                        cv2.imshow('image',img)
                        cv2.waitKey(1)
                        # print("Blink Detected.....!!!!")
        if live_smile and blink_count>=n:
            text_width, text_height = cv2.getTextSize('PASSED', font, 1, 2)[0]
            cv2.putText(img, 'PASSED', (x + w - text_width, y - 5), font, 1, (0, 255, 0), 2)

        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                    )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                    )  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif k == ord('s'):
        first_read=False
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()