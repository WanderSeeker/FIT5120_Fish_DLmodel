# ======================= 自动安装依赖（放在最开头）=======================
import subprocess
import sys
import threading
import time
import os

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 自动检查并安装
install_package("flask")
install_package("torch")
install_package("torchvision")
install_package("pillow")

# ======================= 10 分钟自动关闭程序 =======================
AUTO_STOP_SECONDS = 600  # 600秒 = 10分钟

def auto_stop_server():
    print(f"✅ API 已启动，{AUTO_STOP_SECONDS//60} 分钟后自动关闭")
    time.sleep(AUTO_STOP_SECONDS)
    print("时间到 API 自动关闭")
    os._exit(0)

# 启动后台线程（不影响API运行）
threading.Thread(target=auto_stop_server, daemon=True).start()

# ======================= Flask API 主代码 =======================
from flask import Flask, request
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# 加载模型配置
with open("class_names.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

class_names = meta["class_names"]
img_size = meta["image_size"]
arch = meta["architecture"]

# 模型结构
def build_classifier(in_f, num_c):
    return nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_c)
    )

def build_model(architecture, num_cls):
    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = build_classifier(model.classifier[-1].in_features, num_cls)
        return model

device = torch.device("cpu")
model = build_model(arch, len(class_names)).to(device)
model.load_state_dict(torch.load("fish_disease_mobilenet_v3_small.pt", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 预测接口
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = model(x).argmax(1).item()
    return class_names[pred_idx]

# 主页测试
@app.route("/")
def index():
    return "Fish API 运行中 - 10分钟自动关闭"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)