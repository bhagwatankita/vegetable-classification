# import tensorflow as tf
# # Replace the old imports with these:
# import tensorflow as tf
# from keras import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.src.legacy.preprocessing.image import ImageDataGenerator

# # Config
# IMAGE_SIZE = (150, 150)
# BATCH_SIZE = 32

# # 1. Data Augmentation
# train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_set = train_datagen.flow_from_directory('dataset/train', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
# val_set = val_datagen.flow_from_directory('dataset/validation', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# # 2. Build CNN (Architecture based on your shared images)
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.25),
#     Dense(15, activation='softmax') # 15 categories
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 3. Train
# model.fit(train_set, validation_data=val_set, epochs=10) # Adjust epochs as needed

# # 4. Save
# model.save('vegetable_model.h5')
# print("Model Saved!")



import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# 1. Configuration
IMAGE_SIZE = (224, 224) # MobileNetV2 works best with 224x224
BATCH_SIZE = 32

# 2. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/train', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_set = val_datagen.flow_from_directory('dataset/validation', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# 3. Create the Base Model (Transfer Learning)
# We load MobileNetV2 without the top layer (the classifier)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model (don't train the pre-existing weights)
base_model.trainable = False

# 4. Build your custom classifier on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax') # Your 15 vegetables
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 6. Train
print("Starting Transfer Learning Training...")
model.fit(
    train_set, 
    validation_data=val_set, 
    epochs=20, 
    callbacks=[early_stop]
)

# 7. Save
model.save('vegetable_model.h5')
print("Pro-level model saved successfully!")