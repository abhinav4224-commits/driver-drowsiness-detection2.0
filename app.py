import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Real-time detection using eye strain and head position analysis")

# -----------------------------
# MediaPipe Init
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye, landmarks):
    v1 = np.linalg.norm(landmarks[eye[1]] - landmarks[eye[5]])
    v2 = np.linalg.norm(landmarks[eye[2]] - landmarks[eye[4]])
    h = np.linalg.norm(landmarks[eye[0]] - landmarks[eye[3]])
    return (v1 + v2) / (2.0 * h)

EAR_THRESHOLD = 0.25
DROWSY_TIME = 2

run = st.checkbox("Start Camera")
frame_window = st.image([])
start_time = None
drowsy = False

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in face_landmarks.landmark
            ])

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2

            nose = landmarks[1]
            chin = landmarks[152]
            angle = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))

            if ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= DROWSY_TIME:
                    drowsy = True
            else:
                start_time = None
                drowsy = False

            status = "ALERT"
            color = (0, 255, 0)
            if drowsy or abs(angle) > 15:
                status = "DROWSY"
                color = (0, 0, 255)

            cv2.putText(frame, f"Status: {status}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
