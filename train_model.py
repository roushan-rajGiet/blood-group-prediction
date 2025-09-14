import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
import numpy as np

# --- 1. Data Preparation ---

# Set the path to your dataset
data_dir = 'dataset_blood_group'  # The name of the folder is enough if it's in the same directory

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8 # A+, A-, B+, B-, AB+, AB-, O+, O-

# Create image data generators with augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Load training data from directories
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- 2. Model Architecture (Two-Stage Transfer Learning) ---

# Load the VGG16 model with pre-trained ImageNet weights
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the base model to prevent its weights from being updated
base_model.trainable = False

# Create a new model on top of the pre-trained base
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Add dropout for regularization
    layers.Dense(NUM_CLASSES, activation='softmax') # Output layer for your classes
])

model.summary()

# --- 3. Training Process (Stage 1) ---

# Compile the model with your new classification layers
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting Stage 1: Training top layers...")
# Train only the new layers on top of the frozen VGG16 base
history = model.fit(
    train_generator,
    epochs=10, # Train for a few epochs with frozen layers
    validation_data=validation_generator
)

# --- 4. Training Process (Stage 2: Fine-tuning) ---

# Unfreeze the base model to allow all layers to be trained
base_model.trainable = True

# Re-compile the model with a very low learning rate for fine-tuning
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower learning rate is crucial here
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting Stage 2: Fine-tuning base model layers...")
# Continue training for more epochs
history_fine_tune = model.fit(
    train_generator,
    epochs=40, # Train for more epochs on the fine-tuned model
    validation_data=validation_generator,
    initial_epoch=history.epoch[-1] + 1  # Start from where the previous training left off
)

# --- 5. Save the Model ---

print("\nTraining complete and model saved!")
model.save('blood_group_model.h5') 
