# -*- coding:utf-8 -*-
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 加载模型
model = load_model("coral_model.h5")


# 2. 图像预处理
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# 替换下面的路径为您要分类的图像的路径
image_path = "test-images/Capstone 2023 - AlgaeTraining.jpg"
processed_image = preprocess_image(image_path)

# 3. 模型预测
predictions = model.predict(processed_image)

# 4. 解析预测结果
if predictions[0][0] > 0.5:
    print("该图像被分类为：健康的珊瑚")
else:
    print("该图像被分类为：白化的珊瑚")

