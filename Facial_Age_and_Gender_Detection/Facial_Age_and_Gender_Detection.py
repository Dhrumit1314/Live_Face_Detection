# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:05:30 2024

@author: Dhrumit Patel
"""

"""
Dataset: UTKFace
https://www.kaggle.com/datasets/jangedoo/utkface-new?resource=download
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, Input
from sklearn.model_selection import train_test_split

path = 'data/UTKFace'

images = []
age = []
gender = []

for img in os.listdir(path):
    ages = img.split("_")[0]
    genders = img.split("_")[1]
    img = cv2.imread(str(path) + "/" + str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(np.array(img))
    age.append(np.array(ages))
    gender.append(np.array(genders))

images = np.array(images)

images = images / 255.0
age = np.array(age, dtype=np.int64)
gender = np.array(gender, dtype=np.int64)

np.save('preprocessed_data/images.npy', images)
np.save('preprocessed_data/age.npy', age)
np.save('preprocessed_data/gender.npy', gender)

images = np.load('preprocessed_data/images.npy')
age = np.load('preprocessed_data/age.npy')
gender = np.load('preprocessed_data/gender.npy')

X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)

# Define the model - For Age
age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

age_model.summary()

# TODO: Fit for 10 epochs
with tf.device('/CPU:0'):
    history_age = age_model.fit(X_train_age, y_train_age,
                                epochs=10,
                                validation_data=(X_test_age, y_test_age))  

age_model.save('models/age_model_10epochs.h5')

# Define the model - For Gender
gender_model = Sequential()

gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))

gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gender_model.summary()

# TODO: Fit for 10 epochs
with tf.device('/CPU:0'):
    history_gender = gender_model.fit(X_train_gender, y_train_gender,
                                      epochs=10,
                                      validation_data=(X_test_gender, y_test_gender))

gender_model.save('models/gender_model_10epochs.h5')

# Plot the training and validation accuracy and loss at each epoch - For Age model
loss = history_age.history['loss']
val_loss = history_age.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss for Age model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_age.history['accuracy']
val_acc = history_age.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy for Age model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation accuracy and loss at each epoch - For Gender model
loss = history_gender.history['loss']
val_loss = history_gender.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss for Gender model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_gender.history['accuracy']
val_acc = history_gender.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy for Gender model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Test the model - Gender Model
my_model_gender = load_model('models/gender_model_3epochs.h5')
with tf.device('/CPU:0'):
    predictions = my_model_gender.predict(X_test_gender)
y_pred = (predictions >= 0.5).astype(int)[:, 0]

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

print(f"Accuracy: {accuracy_score(y_true=y_test_gender, y_pred=y_pred)}")

# Confusion matrix
cm = confusion_matrix(y_true=y_test_gender, y_pred=y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# TODO: Test the model - Age Model










