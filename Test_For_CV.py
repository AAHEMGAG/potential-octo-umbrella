import os
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
#Note****** provide it with more practice images
num_classes = 4

IMAGE_FOLDER = "Dataset"
filenames = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_list = []

for file in filenames:
    # Combine folder path with the image file name (e.g., "dataset/img0.png")
    full_path = os.path.join(IMAGE_FOLDER, file)

    # Read the full-color image from the subfolder
    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
    
    if img is not None:
        img_resized = cv2.resize(img, (256, 256))
        image_list.append(img_resized)
        print(f"✅ Successfully loaded {file} from '{IMAGE_FOLDER}'")
    else:
        print(f"❌ Failed to load {file}")


if len(image_list) == 0:
    raise ValueError("No images were loaded! Check your image file locations.")

# Convert list to array & reshape to (num_samples, 256, 256, 3)
X_train = np.array(image_list)

# Inspect RAW max value before dividing by 255
print("Raw pixel max before scaling:", X_train.max())

# Normalize pixel values to range [0.0, 1.0]
X_train = X_train.astype('float32') / 255.0

print("\n--- DATASET SUMMARY ---")
print("Shape of X_train:", X_train.shape) 
print("Min pixel value:", X_train.min()) 
print("Max pixel value:", X_train.max())


y_train = np.array([0, 1, 2, 3])  


# ==========================================
# STEP 2: BUILD & COMPILE THE CNN MODEL
# ==========================================
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),  # Modern Keras input layer syntax
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Binary output (0.0 to 1.0)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================
print("\nStarting training...")
model.fit(X_train, y_train, epochs=20, batch_size=8)

model.save("cow_identifier_model.keras")
print("Model saved successfully as cow_idenentifier_model.keras!")