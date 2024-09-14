from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # โหลดโมเดล YOLOv5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image_url' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    image_url = data['image_url']

    try:
        # ดาวน์โหลดภาพจาก URL
        response = requests.get(image_url)
        response.raise_for_status()  # ตรวจสอบว่า request สำเร็จ

        # เปิดภาพจาก bytes
        img = Image.open(BytesIO(response.content))
        results = model(img)

        # ส่งผลลัพธ์กลับในรูปแบบ JSON
        return jsonify(results.pandas().xyxy[0].to_dict(orient="records"))
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
