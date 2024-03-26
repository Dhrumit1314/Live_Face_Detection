# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:15:10 2024

@author: Dhrumit Patel
"""

import streamlit as st
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np
import datetime

# Load models
face_classifier = cv2.CascadeClassifier('pretrained_haarcascade_classifier/haarcascade_frontalface_default.xml')
emotion_model = load_model('models/emotion_detection_model_50epochs.h5')
age_model = load_model('models/age_model_3epochs.h5')
gender_model = load_model('models/gender_model_3epochs.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Streamlit app
st.title("Live Face Detection")
st.write("This app detects face, emotion, and gender of a person")

start_button = st.button('Start Webcam', key='start')
stop_button = st.button('Stop Webcam', key='stop')

webcam_on = start_button

if webcam_on:
    # Start webcam
    cap = cv2.VideoCapture(0)

    frameST = st.empty()

    while webcam_on and not stop_button:
        ret, frame = cap.read()
        labels = []
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            # Get image ready for prediction
            roi=roi_gray.astype('float')/255.0  # Scaling the image
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

            preds=emotion_model.predict(roi)[0]  # One hot encoded result for 7 classes
            label=class_labels[preds.argmax()]  # Find the label
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            # Gender
            roi_color=frame[y:y+h,x:x+w]
            roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
            gender_label=gender_labels[gender_predict[0]] 
            gender_label_position=(x,y+h+50) # 50 pixels below to move the label outside the face
            cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            # Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
            age = round(age_predict[0,0])
            age_label_position=(x+h,y+h)
            cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        # Display the resulting frame
        frameST.image(frame, channels="BGR")
        
        # Update the webcam status
        webcam_on = not stop_button

    cap.release()
    cv2.destroyAllWindows()


# Write information file of what does this app do
st.write("## Information")

st.title("Facial Emotion Detection, Age, and Gender Recognition with OpenCV")

st.write("""
In this project, a multi-faceted facial analysis system has been implemented, combining state-of-the-art deep learning models with the powerful computer vision library, OpenCV. The system is designed to accurately detect faces in images or video streams and provide insights into facial emotions, age, and gender.
""")

st.header("Project Overview")

st.write("""
- **Face Detection**: Utilizing the robust face detection capabilities of OpenCV, the system efficiently locates and isolates faces within images or video frames. This forms the foundational step for subsequent analyses, ensuring accurate and reliable results.
- **Facial Emotion Detection**: The emotion detection model has been trained to recognize a range of human emotions, including happiness, sadness, anger, surprise, and more. Leveraging a deep learning approach, the model can analyze facial expressions in real-time, providing valuable insights into the emotional states of individuals in the captured media.
- **Age and Gender Recognition**: The age and gender recognition models have been fine-tuned to estimate the age and gender of detected faces. This adds another layer of demographic information, allowing for more comprehensive analysis and applications. The age estimation model provides an approximate age range, while the gender recognition model accurately classifies faces into male or female categories.
- **Integration with OpenCV**: The entire system is seamlessly integrated with OpenCV, a widely-used computer vision library. OpenCV streamlines the image and video processing pipeline, facilitating real-time analysis and enhancing the system's efficiency. The combination of deep learning models and OpenCV ensures a robust and scalable solution for facial analysis tasks.
""")

st.header("Usage")

st.write("""
The information about models is written in the information text file attached along with the GitHub files. The information about the dataset for each training is written in each Python file.

To run the application:
1. Download the models from the link provided in the text file.
2. Save the models in your respective system.
3. Then modify the path of the downloaded models in `live_face_detection.py` file and run it.
""")

st.header("Practical Applications")

st.write("""
This facial analysis system has a wide range of practical applications, including but not limited to:
- Human-computer interaction
- Sentiment analysis
- Audience engagement measurement
- Security and surveillance
- Personalized user experiences
""")

st.header("Future Enhancements")

st.write("""
As technology evolves, there is room for further improvements and enhancements to the system. This may involve fine-tuning the models with additional data, exploring novel architectures, or integrating with other cutting-edge computer vision techniques.
""")

st.header("Conclusion")

st.write("""
In conclusion, the facial emotion detection, age, and gender recognition system presented here showcases the synergy between deep learning models and OpenCV, offering a powerful and versatile tool for understanding and analyzing facial attributes in diverse scenarios.
""")

st.markdown(f"© {datetime.datetime.now().year} Dhrumit Patel. All rights reserved.")