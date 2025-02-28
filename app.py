from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import AdelaiDet.model_pred as md
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import random
from AdelaiDet.model_pred.cResNet import ResNetModel

app = Flask(__name__)

# 设置上传文件的存储路径和允许的文件类型
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = md.Diagnose_model()
class_names = ['Spacing', 'Normal', 'Mild', 'Moderate', 'Severe']

# 判断文件类型是否有效
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_pil_image(base64_data):
    # 去除 base64 数据前缀（如 "data:image/jpeg;base64,"）
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    # 解码 base64 字符串
    img_data = base64.b64decode(base64_data)
    # 将字节数据加载到 BytesIO
    img = Image.open(BytesIO(img_data))
    return img


def draw_mask(image, masks, alpha=0.3):
    # 创建一个透明背景的图像，大小与原图相同
    output = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)  # 4 通道 (RGBA)

    # 遍历每个 mask 并绘制到透明背景上
    for idx in range(len(masks)):
        # 生成随机颜色
        color = [random.randint(0, 255) for _ in range(3)]
        # 当前 mask
        mask = masks[idx].astype(np.uint8)

        # 创建彩色蒙版，并设置透明度
        color_mask = np.zeros_like(output, dtype=np.uint8)
        for i in range(3):
            color_mask[:, :, i] = mask * color[i]

        # 设置透明度
        alpha_mask = mask.astype(np.uint8) * alpha * 255  # 透明度范围为 0 到 255
        color_mask[:, :, 3] = alpha_mask  # 设置 alpha 通道

        # 将当前蒙版叠加到输出图像
        output = np.maximum(output, color_mask)

    # 返回透明背景的蒙版图像
    return Image.fromarray(output, 'RGBA')

@app.route('/')
def index():
    return render_template('index.html')

# 处理图像的接口
@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()  # 获取请求数据
    image_path = data['image_path']
    flag = data['flag']
    img = base64_to_pil_image(image_path)
    # 调用模型获取框和点
    pred_score, pred_classes, points, boxes, masks = model.run_model(img, flag)
    pred_score = pred_score.tolist()[0]
    for i, score in enumerate(pred_score):
        pred_score[i] = round(score, 3)*100
    pred_class = pred_classes.cpu().tolist()[0]
    pred_class = class_names[pred_class]
    filtered_point_list = [point for point in points if point != [0, 0]]

    # 生成透明背景的彩色蒙版
    new_masks = draw_mask(img, masks)

    # 将图像转换为 base64
    buffered = BytesIO()
    new_masks.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "boxes": boxes,
        "points": filtered_point_list,
        "pred_score": pred_score,
        "pred_class": pred_class,
        "mask_url": f"data:image/png;base64,{mask_base64}"
    })


if __name__ == '__main__':
    app.run(debug=True)