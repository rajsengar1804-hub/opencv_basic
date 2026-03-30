import cv2
img=cv2.imread("image.png")
cv2.imshow("this is human image",img)
cv2.line(img,(0,50),(100,150),(0,255,0),3)
cv2.rectangle(img,(0,40),(100,120),(0,100,0),3)
cv2.circle(img,(0,20),20,(0,255,0),-1)
cv2.imshow("this is human image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()