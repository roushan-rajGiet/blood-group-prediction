import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. Data Preparation ---

data_dir = 'dataset_blood_group'   # root folder with 8 subfolders

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8  # A+, A-, B+, B-, AB+, AB-, O+, O-

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

#  validation: only rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print("Class indices:", train_generator.class_indices)

# --- 2. Model Architecture (VGG16 + custom head) ---

base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model.trainable = False  # Stage 1: freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)
model.summary()

# --- callbacks ---

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_blood_group_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1)
]

# --- 3. Stage 1: train top layers only ---

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting Stage 1: Training top layers...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 4. Stage 2: fine-tune last VGG16 block ---

base_model.trainable = True

set_trainable = False
for layer in base_model.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    layer.trainable = set_trainable

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting Stage 2: Fine-tuning last VGG16 block...")
history_finetune = model.fit(
    train_generator,
    epochs=30,  # callbacks will stop earlier if no improvement
    validation_data=validation_generator,
    callbacks=callbacks,
    initial_epoch=len(history.epoch)
)

print("\nTraining complete. Best model saved as best_blood_group_model.h5")
