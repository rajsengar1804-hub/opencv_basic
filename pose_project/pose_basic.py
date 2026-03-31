import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        print("Nose:", nose.x, nose.y)
        print("Left Shoulder:", left_shoulder.x, left_shoulder.y)
        print("Right Shoulder:", right_shoulder.x, right_shoulder.y)
        print("----------------------")

        h, w, c = frame.shape   # get frame size

        lm = results.pose_landmarks.landmark
        
        nose = lm[0]
        left_shoulder = lm[11]
        right_shoulder = lm[12]

        # convert to pixels
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)

        print("Nose Pixel:", nose_x, nose_y)
        print("Left Shoulder Pixel:", ls_x, ls_y)
        print("Right Shoulder Pixel:", rs_x, rs_y)
        print("-------------------")
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()