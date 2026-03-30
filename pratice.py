import cv2
img=cv2.imread('image.png')# read into array form 
img[0:100,0:80]=[0,255,0]
cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
