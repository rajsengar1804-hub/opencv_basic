import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow("frame",frame)
    # draw rectangle
    cv2.rectangle(frame, (50,50), (300,200), (255,0,0), 3)

    # draw circle
    cv2.circle(frame, (500,100), 40, (0,0,255), -1)

    # write text
    cv2.putText(frame, "AI is running",
                (50,400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()