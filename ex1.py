# Face recognization
import cv2
import os

alg = "haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)

path = "dataset"

if not os.path.isdir(path):
	os.mkdir(path)

(width,height) = (100,100)
count = 0

while count<100:
	count+=1
	print(count)
	_,img = cam.read()
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = haar.detectMultiScale(grayImg,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
		onlyFace = grayImg[y:y+h,x:x+w]
		resizeImg = cv2.resize(onlyFace, (width,height))
		cv2.imwrite("%s/%s.jpg"%(path,count), resizeImg)
		
	cv2.imshow("faceDection", img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
print("Sucessfully colleted dataset")
cam.release()
cv2.destroyAllWindows()