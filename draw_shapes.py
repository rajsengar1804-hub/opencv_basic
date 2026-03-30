import cv2
img=cv2.imread('image.png')
# cv2.imshow('my image',img)
#draw a line on images
# cv2.line(img,(50,50),(200,200),(0,100,0),3)
#draw a rectangle
cv2.rectangle(img, (50,100), (250,250), (255,0,0), -1)
cv2.circle(img,(0,50),50,(0,255,0),3)#-1 is used to filled the circle ,rectangle 
cv2.putText(img, "AI Elderly Guardian",
            (0,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2)
cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



