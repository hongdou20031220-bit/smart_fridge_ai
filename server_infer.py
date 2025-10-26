from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os
import json
from datetime import datetime, timedelta


# 1️⃣ 创建 Flask 应用
app = Flask(__name__)

# 2️⃣ 加载预训练模型（只加载一次）
model = MobileNetV2(weights='imagenet')

# ======================
# 🍎 保质期管理逻辑
# ======================

def get_expiry_days(fruit_name):
    # 定义常见水果的保质期（天）
    expiry_dict = {
        "apple": 7,
        "banana": 3,
        "orange": 10,
        "strawberry": 2,
        "pomegranate": 10
    }
    return expiry_dict.get(fruit_name.lower(), 5)  # 默认5天

def save_record(fruit_name):
    record_path = "data/expiry_data.json"
    if not os.path.exists("data"):
        os.makedirs("data")

    expiry_days = get_expiry_days(fruit_name)
    record = {
        "fruit": fruit_name,
        "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expiry_date": (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d %H:%M:%S"),
        "expiry_days": expiry_days
    }

    # 读取旧数据并追加
    try:
        with open(record_path, "r") as f:
            data = json.load(f)
    except:
        data = []
    
    data.append(record)

    # 写入新数据
    with open(record_path, "w") as f:
        json.dump(data, f, indent=4)

# 3️⃣ 首页路由：测试服务器是否运行
@app.route('/')
def home():
    return "Smart Fridge AI Server with Image Recognition is Running!"

# 4️⃣ 识别接口：接收图片并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    # 打开图片并转换为模型输入格式
    image = Image.open(file.stream).convert('RGB')
    image = image.resize((224, 224))  # MobileNetV2 要求 224x224
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 维度
    img_array = preprocess_input(img_array)  # 标准化到模型输入范围

    # 模型预测
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]  # 取前三个结果

    # 格式化输出结果
    results = [{'class': c, 'description': d, 'confidence': float(s)} for (c, d, s) in decoded]

    # 取最有可能的预测结果
    main_prediction = decoded[0][1]  # 例如 'apple'

    # 保存记录到数据库
    save_record(main_prediction)

    return jsonify({'predictions': results})
from flask import render_template

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream)
            img = img.resize((224, 224))
            x = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(x)
            top3 = decode_predictions(preds, top=3)[0]
            result = [f"{desc}: {prob:.2%}" for (_, desc, prob) in top3]
            
            main_prediction = top3[0][1]  # 取最可能的预测结果
            save_record(main_prediction)

            return render_template('upload.html', result=result)
    return render_template('upload.html')

# 给ESP的最新记录，因为历史记录不是ESP请求的
@app.route('/latest', methods=['GET'])
def latest_record():
    record_path = "data/expiry_data.json"
    if not os.path.exists(record_path):
        return jsonify({"message": "no data"})

    with open(record_path, "r") as f:
        data = json.load(f)

    if len(data) == 0:
        return jsonify({"message": "no records"})

    latest = data[-1]  # 取最后一个记录
    return jsonify(latest)

# 5️⃣ 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051, debug=True)
