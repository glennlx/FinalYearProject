import cv2
import os
cwd = os.getcwd()
cwd=cwd.replace("\\","/")
face_cascade = cv2.CascadeClassifier(cwd+'/HaarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cwd+'/HaarCascades/haarcascade_eye.xml')

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('Live Stream : ', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

