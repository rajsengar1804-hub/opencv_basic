import cv2
import mediapipe as mp

# initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# function to get all landmark pixel coordinates
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

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()