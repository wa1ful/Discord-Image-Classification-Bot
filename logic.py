import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Функция для предсказания
def detect(image_path):
  np.set_printoptions(suppress=True)
  model = tf.keras.models.load_model("keras_model.h5", compile=False)
  
  with open("labels.txt", "r", encoding="utf-8") as f:
      class_names = [line.strip() for line in f.readlines()]
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  image = Image.open(image_path).convert("RGB")
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
  image_array = np.asarray(image)
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  
  return class_name, confidence_score

def check_required_files():
    required_files = ["keras_model.h5", "labels.txt"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files