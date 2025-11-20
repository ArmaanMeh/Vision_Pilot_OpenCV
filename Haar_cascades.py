face_cascade = cv2.CascadeClassifier("opencvCW\OPENCV CW\Haar_Cadcades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("opencvCW\OPENCV CW\Haar_Cadcades\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencvCW\OPENCV CW\Haar_Cadcades\haarcascade_smile.xml")

   #Face Detection
faces = face_cascade.detectMultiScale(img,1.1,5)
for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"Face detected",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,0),2)
    
    #eye detection
eye = eye_cascade.detectMultiScale(img,1.1,5)
for x,y,w,h in eye:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"Eyes detected",(x-40,y-40),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,0),2)
    
    #Smile detection
smile = smile_cascade.detectMultiScale(img,1.1,5)
for x,y,w,h in smile:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"Smile detected",(x-40,y-40),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,0),2)

