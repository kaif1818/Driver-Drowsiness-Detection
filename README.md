# Driver Drowsiness Detection

A real-time driver drowsiness detection system using computer vision and facial landmark analysis. This project monitors a driver's alertness by detecting closed eyes and yawning using the webcam. If drowsiness is detected, a beep alert is triggered to warn the driver, promoting safer driving.

---

## **Features**
- Detects eye closure using Eye Aspect Ratio (EAR)
- Detects yawning using Mouth Aspect Ratio (MAR)
- Real-time webcam monitoring
- Beep alert when driver shows signs of fatigue
- Lightweight and fast using HOG face detection

---

## **Tech Stack**
- **Python 3.x**
- **OpenCV** – for video capture and image processing
- **face_recognition** – for face detection and landmark extraction
- **NumPy** – for numerical calculations
- **SciPy** – for Euclidean distance calculations
- **winsound** – to generate beep alerts (Windows only)

---

## **How It Works**

1. **Capture Webcam Frames:**  
   The system captures frames from the webcam using OpenCV.

2. **Face Detection:**  
   Each frame is converted to RGB and faces are detected using the HOG model for speed.

3. **Facial Landmarks Extraction:**  
   Facial landmarks are detected using the `face_recognition` library, focusing on the eyes and mouth.

4. **Calculate EAR and MAR:**  
   - **Eye Aspect Ratio (EAR):** Measures the openness of eyes. If below threshold → eyes closed.  
   - **Mouth Aspect Ratio (MAR):** Measures mouth opening. If above threshold → yawning detected.

5. **Score System:**  
   - Each frame that detects drowsiness increases the score.  
   - Frames without drowsiness decrease the score.  
   - If the score exceeds a defined threshold, a beep alert is triggered.

6. **Real-Time Feedback:**  
   - Score is displayed on the webcam feed.  
   - Alerts the driver if signs of fatigue are detected.

---

## **Usage**

1. Clone the repository:

```bash
git clone https://github.com/kaif1818/Driver-Drowsiness-Detection.git
