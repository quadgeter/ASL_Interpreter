from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import joblib
import numpy as np

# Flask setup
app = Flask(__name__)

# Load trained model + label encoder
model = joblib.load("asl_svm_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = ""

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            wrist = landmarks.landmark[0]
            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            if len(row) == 63:
                X_input = np.array(row).reshape(1, -1)
                pred_idx = model.predict(X_input)[0]
                prediction = label_encoder.inverse_transform([pred_idx])[0]

                # Draw the prediction on the frame
                cv2.putText(frame, f'Prediction: {prediction}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
