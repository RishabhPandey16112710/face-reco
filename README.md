# face-reco
import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import time

st.title('Facial Emotion Recognition')

run = st.button('Start Camera')

if run:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    # For streamlit cloud, you'd need to use a different approach
    # as direct webcam access might be limited
    
    st.write("Press Stop when done")
    stop = st.button('Stop')
    
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.write("Can't receive frame")
            break
            
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Display emotion on frame
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Convert BGR to RGB for streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
        except Exception as e:
            st.write(f"Error: {e}")
            
        time.sleep(0.1)
        
    cap.release()
