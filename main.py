import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import warnings
import winsound  # For beep alert
warnings.filterwarnings("ignore")

# ---------------------------
# Helper functions
# ---------------------------

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# ---------------------------
# Fast face processing
# ---------------------------
def process_frame(frame):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.7

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use HOG for fast CPU detection
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    eye_flag = False
    mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]

        # Eyes
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            if ear < EYE_AR_THRESH:
                eye_flag = True
        # Mouth
        if 'mouth' in landmarks:
            mouth = np.array(landmarks['mouth'])
            mar = mouth_aspect_ratio(mouth)
            if mar > MOUTH_AR_THRESH:
                mouth_flag = True

    return eye_flag, mouth_flag

# ---------------------------
# Real-time webcam detection
# ---------------------------
video_cap = cv2.VideoCapture(0)
count = 0
score = 0
ALERT_SCORE = 10  # Score threshold to trigger beep

print("\nStarting webcam... Press 'q' to quit.\n")

while True:
    success, frame = video_cap.read()
    if not success:
        break

    # Reduce frame size for speed
    small_frame = cv2.resize(frame, (320, 240))
    count += 1

    # Process every 3rd frame
    if count % 3 == 0:
        eye_flag, mouth_flag = process_frame(small_frame)

        if eye_flag or mouth_flag:
            score += 1
        else:
            score -= 1
            if score < 0:
                score = 0

    # Beep alert if score exceeds threshold
    if score > ALERT_SCORE:
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms

    # Display score
    font = cv2.FONT_HERSHEY_COMPLEX
    text = f"Score: {score}"
    cv2.putText(frame, text, (10, frame.shape[0]-20), font, 1, (0,0,255), 2, cv2.LINE_AA)

    # Show video
    cv2.imshow("Drowsiness Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
