from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io, os
import tensorflow as tf

app = Flask(__name__)

# ✅ โหลด TFLite Model
TFLITE_PATH = "classifier_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# ดึง input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ไม่มีไฟล์ที่อัปโหลด"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ไม่ได้เลือกไฟล์"}), 400

    try:
        # โหลดและแปลงภาพ
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        # รันผ่าน TFLite
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

        confidence = float(pred)
        is_solution = confidence > 0.5
        intensity = int(np.mean(img_array) * 255)

        return jsonify({
            "is_solution": is_solution,
            "confidence": confidence,
            "intensity": intensity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
