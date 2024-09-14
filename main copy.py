from flask import Flask, request, jsonify,render_template
import torch
from PIL import Image

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # โหลดโมเดล YOLOv5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file)
    results = model(img)
    return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == '__main__':
    # app.run()
    app.run(port=5000)
