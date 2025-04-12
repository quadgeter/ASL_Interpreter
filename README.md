# ASL Interpreter – Real-Time American Sign Language Detection

This is a real-time American Sign Language (ASL) interpreter built using Python, Flask, MediaPipe, OpenCV, and a machine learning classifier (SVM). It captures hand gestures from your webcam and classifies them as letters or numbers from the ASL alphabet using landmark-based hand tracking.

---

## Features

- Real-time webcam gesture detection using MediaPipe
- Classifies ASL letters and numbers (A–Z, 0–9)
- Trained on preprocessed 3D hand landmark data
- Clean, responsive web interface with live video stream
- Extensible model and data augmentation pipeline

---

## Quickstart

### Requirements

- python 3.8+ 
- flask
- opencv-python
- mediapipe
- scikit-learn
- joblib
- pandas

