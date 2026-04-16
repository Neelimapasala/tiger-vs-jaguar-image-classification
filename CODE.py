# Tiger Vs Jaguar Image Classification

Team:-

Name: Pasala Neelima, Devika P 

Reg No: 24MDT1064, 24MDT1057 
# Abstract
Wildlife image classification plays a crucial role in biodiversity monitoring and conservation, yet distinguishing between visually similar species such as tigers and jaguars remains a challenging task. Manual identification is often time-consuming, error-prone, and impractical for large-scale deployments such as camera trap networks. To address this, our project explores automated classification using both deep learning and classical machine learning approaches. We employ a transfer learning pipeline based on ResNet50, a state-of-the-art Convolutional Neural Network, trained on augmented Kaggle dataset images to capture complex spatial features. In parallel, handcrafted feature extraction is performed using texture descriptore and color statistics, combined with Random Forest classification, incorporating Region of Interest segmentation via Otsu thresholding. The dual approach enables comparison between end-to-end deep learning and feature-engineered machine learning methods. Dataset images are split into 75% training and 15% validation subsets, and evaluation metrics include accuracy, precision, F1-score, and confusion matrix analysis. Results highlight the effectiveness of CNN-based transfer learning in handling fine-grained interspecies confusion while demonstrating the value of classical approaches for smaller datasets. This work represents the first focused comparative study on tiger versus jaguar classification, with implications for AI-driven wildlife monitoring, ecological research, and automated camera trap applications.
Problem Statement- Manual identification of tigers and jaguars from camera trap images is slow,
 error-prone, and challenging due to their high visual similarity and varying environmental
 conditions. Existing automated systems rarely focus on this specific fine-grained classification,
 leading to frequent misclassification. This project aims to develop a deep learning model to
 accurately distinguish between the two species, improving efficiency in wildlife monitoring and
 conservation.
Step:1 import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 Step2:Calling dataset
 
  dataset/ ├── tiger/ └── jaguar/
dataset_dir = "dataset" 
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32
 Step3: Data Augmentation & Preprocessing
 
 
 * Resizing
 * Normalization
 * Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.15  
)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
 Step5: Sample images
def show_sample_images(generator):
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((images[i] + 1) / 2)  
        plt.title(f"Class: {list(generator.class_indices.keys())[np.argmax(labels[i])]}")
        plt.axis("off")
    plt.show()

show_sample_images(train_generator)
Step 6 : Build ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)        # Reduce dimensions
x = Dropout(0.3)(x)                    # Prevent overfitting
predictions = Dense(2, activation='softmax')(x)  # 2 classes: Tiger, Jaguar

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)
val_generator.reset()
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix – Tiger vs Jaguar")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    pred = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
predict_image("sample1.png")
predict_image("sample2.png")
predict_image("sample3.png")
predict_image("sample4.png")
predict_image("sample5.png")
predict_image("sample6.png")
predict_image("sample7.png")
predict_image("sample8.png")
predict_image("sample9.png")
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

