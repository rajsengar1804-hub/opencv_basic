import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_landmark_list(results, frame):
    h, w, c = frame.shape
    landmark_list = []

    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            px, py = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, px, py])

    return landmark_list


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

import math

def find_distance(p1, p2):
    x1, y1 = p1[1], p1[2]
    x2, y2 = p2[1], p2[2]
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

with mp_pose.Pose() as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        landmarks = get_landmark_list(results, frame)

        if len(landmarks) != 0:
            print("Nose:", landmarks[0])
            print("Left Shoulder:", landmarks[11])
            print("Right Shoulder:", landmarks[12])
            print("Left Hip:", landmarks[23])
            print("Right Hip:", landmarks[24])
            print("Left Knee:", landmarks[25])
            print("Right Knee:", landmarks[26])
            print("Left Ankle:", landmarks[27])
            print("Right Ankle:", landmarks[28])
            print("--------------------------")

            nose = landmarks[0]
            left_ankle = landmarks[27]
            body_height = find_distance(nose, left_ankle)
            print("Body Height:", int(body_height))

            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            shoulder_width = find_distance(l_shoulder, r_shoulder)
            print("Shoulder Width:", int(shoulder_width))

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()