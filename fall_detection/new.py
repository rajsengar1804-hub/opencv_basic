import cv2
import mediapipe as mp
import numpy as np
import math
from dataclasses import dataclass
from enum import Enum

# =========================================================
# ENUMS
# =========================================================

class PostureState(Enum):
    STANDING="Standing"
    SITTING="Sitting"
    LYING="Lying"
    BENDING="Bending"
    UNKNOWN="Unknown"

# =========================================================
# POSE METRICS
# =========================================================

@dataclass
class PoseMetrics:
    body_height: float
    torso_angle: float
    left_knee: float
    right_knee: float

# =========================================================
# POSE ANALYZER
# =========================================================

class PoseAnalyzer:

    def angle(self,a,b,c):
        a=np.array(a); b=np.array(b); c=np.array(c)
        ba=a-b; bc=c-b
        cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        return np.degrees(np.arccos(np.clip(cos,-1,1)))

    def torso_angle(self,shoulder_mid,hip_mid):
        vector=np.array(hip_mid)-np.array(shoulder_mid)
        vertical=np.array([0,1])
        cos=np.dot(vector,vertical)/(np.linalg.norm(vector)+1e-6)
        angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
        return min(angle,180-angle)

    def extract_metrics(self,lm,w,h):
        def P(id): return np.array([lm[id].x*w, lm[id].y*h])

        nose=P(0)
        ls=P(11); rs=P(12)
        lh=P(23); rh=P(24)
        lk=P(25); rk=P(26)
        la=P(27); ra=P(28)

        shoulder_mid=(ls+rs)/2
        hip_mid=(lh+rh)/2
        ankle_mid=(la+ra)/2

        body_height=np.linalg.norm(nose-ankle_mid)
        torso_angle=self.torso_angle(shoulder_mid,hip_mid)
        left_knee=self.angle(lh,lk,la)
        right_knee=self.angle(rh,rk,ra)

        return PoseMetrics(body_height,torso_angle,left_knee,right_knee)

# =========================================================
# POSTURE CLASSIFIER (FINAL FIXED)
# =========================================================

class PostureClassifier:
    def __init__(self):
        self.standing_height=None

    def classify(self,m:PoseMetrics):

        avg_knee=(m.left_knee+m.right_knee)/2

        # Auto calibration (first few frames)
        if self.standing_height is None:
            self.standing_height=m.body_height
            return PostureState.STANDING

        height_ratio=m.body_height/self.standing_height

        # LYING → body collapses
        if height_ratio<0.55:
            return PostureState.LYING

        # SITTING
        if avg_knee<140 and m.torso_angle<35:
            return PostureState.SITTING

        # BENDING
        if 35<m.torso_angle<65:
            return PostureState.BENDING

        # STANDING
        if avg_knee>150 and m.torso_angle<35:
            return PostureState.STANDING

        return PostureState.UNKNOWN

# =========================================================
# MAIN
# =========================================================

def main():
    cap=cv2.VideoCapture(0)
    mp_pose=mp.solutions.pose
    mp_draw=mp.solutions.drawing_utils

    analyzer=PoseAnalyzer()
    classifier=PostureClassifier()

    with mp_pose.Pose() as pose:
        while True:
            ret,frame=cap.read()
            if not ret: break

            h,w,_=frame.shape
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results=pose.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

                metrics=analyzer.extract_metrics(results.pose_landmarks.landmark,w,h)
                posture=classifier.classify(metrics)

                cv2.putText(frame,f"{posture.value}",(30,60),
                            cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            cv2.imshow("Posture Detection",frame)
            if cv2.waitKey(1)==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()