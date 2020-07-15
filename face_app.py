import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("test_2.jpg")

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_img = cv2.GaussianBlur(gray_img,(21,21),0)

face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5)

for x,y,w,h in face:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#resized = cv2.resize(img,int(img.shape[1]/2),int(img.shape[0]/2))

cv2.imshow("Gray",img)

cv2.waitKey(0)

cv2.destroyAllWindows()
