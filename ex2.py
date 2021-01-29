# Face recognization
import cv2,os,imutils
import numpy as np
hars = "haarcascade_frontalface_default.xml"
dataset = "dataset"
(images,labels,names,id) = ([],[],{},0)

for (subdir,dirs,files) in os.walk(dataset):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(dataset,subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath+'/'+filename
			label = id
			images.append(cv2.imread(path,0))
			labels.append(int(label))
			# print(labels)
		id+=1

(width,height) = (130,100)

(images,labels) = [np.array(lis) for lis in [images,labels]]
# print((images,labels))

# model =  cv2.face.LBPHFaceRecognizer_create()
model =  cv2.face.LBPHFaceRecognizer_create()

model.train(images,labels)

print("training completed")

# Testing 
import cv2
import time
time.sleep(1)
cam = cv2.VideoCapture(0)
cnt=1

while True:
	_,frame = cam.read()
	grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face_cascad = cv2.CascadeClassifier(hars)
	faces = face_cascad.detectMultiScale(grayImg,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),3)
		face = grayImg[y:y+h,x:x+w]
		face_resize = cv2.resize(face, (width,height))

		pred = model.predict(face_resize)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)

		if pred[1]<500:
			config = int( 100 * (1 - (pred[1])/400) )
			cv2.putText(frame, "%s-%.0f"% (names[pred[0]],config), (x-10,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0),2)
			print(names[pred[0]])
		else:
			cnt +=1
			cv2.putText(frame, "unknown",(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
			if cnt >100:
				print("unknown face")
				cv2.imwrite("unknownface.jpg", frame)
				cnt =0

	cv2.imshow("OpenCV", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()