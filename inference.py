from config import config
from main import test_transform

import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image
import torch


model_path = 'models/best_age_estimation_model'
model = torch.load(model_path)
model = model.to(config['device'])
model.eval()

def inference(image_path, transform, model, face_detection=False):
  if face_detection:
    img = face_recognition.load_image_file(image_path)
    top, right, bottom, left = face_recognition.face_locations(img)[0]
    img_crop = img[top:bottom, left:right]
    img_crop = Image.fromarray(img_crop)
  else:
    img_crop = Image.open(image_path).convert('RGB')

  img_tensor = transform(img_crop).unsqueeze(0)
  with torch.inference_mode():
    preds = model(img_tensor.to(config['device'])).item()

  return preds, img_crop


preds, img = inference('test.jpg',
                       test_transform, model, face_detection=True)

print(f'{preds:.2f}')
plt.imshow(img)


# Load a random image from a folder
folder_image_path = 'data/UTKFace/'
image_files = os.listdir(folder_image_path)

rand_idx = random.randint(0, len(image_files))
test_image_path = os.path.join(folder_image_path, image_files[rand_idx])
predicted_age, image = inference(test_image_path, test_transform, model)

real_age = image_files[rand_idx].split('_')[0]
print(f"Real: {real_age}, Predicted: {predicted_age:.2f}")
plt.imshow(image)


# Load a random image from a csv file
csv_file_path = 'data/test_set.csv'
df = pd.read_csv(csv_file_path)

rand_idx = random.randint(0, df.shape[0])
test_image_name = df.iloc[rand_idx].image_name
test_image_path = os.path.join(folder_image_path, test_image_name)
predicted_age, image = inference(test_image_path, test_transform, model)

real_age = test_image_name.split('_')[0]
print(f"Real: {real_age}, Predicted: {predicted_age:.2f}")
plt.imshow(image)
