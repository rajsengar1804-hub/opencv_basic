import cv2

# Read image from file
img = cv2.imread("image.png")
img[0:100, 0:100] = [0, 255, 0]
cv2.imshow("My Image", img)


cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()
print(img)
print(img.shape)

