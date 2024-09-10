# USAGE
# python detect_faces.py --face cascades/haarcascade_frontalface_default.xml --image images/obama.png

# import the necessary packages
import cv2
import imutils

detector_face = cv2.CascadeClassifier("cascades\\haarcascade_frontalface_default.xml")

detector_eye = cv2.CascadeClassifier("cascades\\haarcascade_eye.xml")


camera = cv2.VideoCapture(0)

while True:
    (grabbed,frame) = camera.read()

    if not grabbed:
        break


    frame = imutils.resize(frame,width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = detector_face.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faceRects:

        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0, 0),5)



        eyeRects = detector_eye.detectMultiScale(gray)
        for (ex,ey,ew,eh) in eyeRects:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

