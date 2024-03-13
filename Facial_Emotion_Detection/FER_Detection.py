# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:22:48 2024

@author: Dhrumit Patel
"""

"""
Dataset: FER-2013
https://www.kaggle.com/datasets/msambare/fer2013
"""

"""
NOTE: PLEASE CHANGE THE PATHS ACCORDING TO YOUR SYSTEM AFTER DOWNLOADING THE FILES
"""

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    color_mode='grayscale',
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              color_mode='grayscale',
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=True)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_generator.__next__()

import random
i = random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()

# Define the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from keras.utils import plot_model
plot_model(model, show_dtype=True, show_layer_names=True)

train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs//batch_size,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs//batch_size)

model.save('models/emotion_detection_model_50epochs.h5')

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from keras.models import load_model
my_model = load_model('models/emotion_detection_model_50epochs.h5',compile=False)

# Generate a batch of images
test_img, test_lbl = validation_generator.__next__()
predictions = my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix
print(f"Accuracy: {accuracy_score(y_true=test_labels, y_pred=predictions)}")

cm = confusion_matrix(y_true=test_labels, y_pred=predictions)
cm

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
original_label = class_labels[test_labels[n]]
predicted_label = class_labels[predictions[n]]
plt.imshow(image[:, :, 0], cmap='gray')
plt.title(f"Original Label: {original_label} | Predicted Label: {predicted_label}")
plt.axis("off")
plt.show()
