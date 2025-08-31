import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ===============================
# กำหนดค่าพื้นฐาน
# ===============================
DATASET_DIR = "dataset"   # ตรงนี้ใส่ path dataset ของคุณ
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUTPUT = "solution_classifier_tflite"

os.makedirs(MODEL_OUTPUT, exist_ok=True)

# ===============================
# เตรียม Data Generator
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # ✅ multi-class
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # ✅ multi-class
    subset="validation"
)

num_classes = len(train_generator.class_indices)
print("พบ Classes:", train_generator.class_indices)

# ===============================
# สร้างโมเดล CNN
# ===============================
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")  # ✅ multi-class output
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# เทรนโมเดล
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ===============================
# บันทึกโมเดลปกติ (.h5)
# ===============================
h5_path = os.path.join(MODEL_OUTPUT, "classifier.h5")
model.save(h5_path)
print(f"✅ โมเดลถูกบันทึกที่ {h5_path}")

# ===============================
# แปลงเป็น TensorFlow Lite
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_OUTPUT, "classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ โมเดล TFLite ถูกบันทึกที่ {tflite_path}")
