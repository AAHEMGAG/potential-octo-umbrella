import os
import re
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import shuffle

num_classes = 3
IMAGE_FOLDER = "Dataset_1"

def get_file_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# Get filenames sorted numerically (img0, img1, ..., img500, img501, ..., img850)
filenames = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
filenames = sorted(filenames, key=get_file_number)

image_list = []

for file in filenames:
    full_path = os.path.join(IMAGE_FOLDER, file)
    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
    
    if img is not None:
        img_resized = cv2.resize(img, (128, 128))
        img_scaled = img_resized.astype('float32') / 255.0
        image_list.append(img_scaled)

X_train = np.array(image_list, dtype='float32')

# Total: 851 images (0 through 500 = 501 images; 501 through 850 = 350 images)
num_class_0 = 500
num_class_1 = 500
num_class_2 = 500

y_train = np.concatenate([
    np.zeros(num_class_0, dtype=int), 
    np.ones(num_class_1, dtype=int),
    np.full(num_class_2, 2, dtype=int)  
])

print("Dataset successfully loaded and aligned!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Class 0 count:", np.sum(y_train == 0))
print("Class 1 count:", np.sum(y_train == 1))
print("Class 2 count:", np.sum(y_train == 2))

print("\n--- DATASET SUMMARY ---")
print("Shape of X_train:", X_train.shape) 
print("Min pixel value:", X_train.min()) 
print("Max pixel value:", X_train.max())

print(y_train)



# ==========================================
# STEP 2: BUILD & COMPILE THE CNN MODEL
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.2),  # Rotates image up to 10%
    layers.RandomZoom(0.2),      # Zooms in/out up to 10%
    layers.RandomContrast(0.2),  # Adjusts contrast slightly
    layers.RandomBrightness(0.2)   # Adjusts brightness slightly
], name="data_augmentation")

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Modern Keras input layer syntax

    data_augmentation,

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

X_shuffled, y_shuffled = shuffle(X_train, y_train, random_state=42)
# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================
print("\nStarting training...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_shuffled, y_shuffled, epochs=40, batch_size=32, validation_split=0.2, callbacks=[early_stop])

model.save("cow_identifier_model.keras")
print("Model saved successfully as cow_identifier_model.keras!")