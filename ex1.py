# face dataset ceration
import cv2,os,imutils
import numpy as np
import time
time.sleep(1)

hars = "haarcascade_frontalface_default.xml"
dataset = "dataset"
name = "jyothi"
# name = "vijaya"
# name = "vidhydhar"

path = os.path.join(dataset,name)
# print(path)

if not os.path.isdir(path):
	os.makedirs(path)

(width,height) = (130,100)

face_cascad = cv2.CascadeClassifier(hars)

cam = cv2.VideoCapture(0)
count=0

while count<100:
	count+=1
	print(count)
	_,frame = cam.read()
	grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascad.detectMultiScale(grayImg,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),3)
		face = grayImg[y:y+h,x:x+h]
		face_resize = cv2.resize(face, (width,height))
		cv2.imwrite("%s/%s.jpg"%(path,count), face_resize)

	cv2.putText(frame, "Count: "+str(count) , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
	cv2.imshow("OpenCV", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
print(f"Sucessfully colleted dataset of {name}")
cam.release()
cv2.destroyAllWindows()