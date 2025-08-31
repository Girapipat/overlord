import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# === โหลดโมเดล TFLite ===
MODEL_PATH = "classifier.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# เอา metadata ของโมเดลมา
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ตั้งค่า Flask app
app = Flask(__name__, static_url_path="", static_folder="static", template_folder="templates")

# อ่าน input shape ของโมเดล เช่น (1, 224, 224, 3)
INPUT_SHAPE = input_details[0]['shape']
IMG_SIZE = (INPUT_SHAPE[1], INPUT_SHAPE[2])

# === Preprocess function ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        image = Image.open(file).convert("RGB")
        input_data = preprocess_image(image)

        # ใส่ข้อมูลเข้า interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # คาดว่าเป็น multi-class → softmax
        predicted_class = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))

        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
