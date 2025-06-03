import os
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
      self.root_dir = root_dir
      self.csv_file = csv_file
      self.transform = transform
      self.data = pd.read_csv(self.csv_file)
      self.gender_dict = {
          'Male': 0,
          'Female': 1
      }
      self.ethnicity_dict = {
          'White': 0,
          'Black': 1,
          'Asian': 2,
          'Indian': 3,
          'Others': 4
      }

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      sample = self.data.iloc[idx, :]
      image_name = sample.image_name
      image = Image.open(os.path.join(self.root_dir, image_name))
      image = self.transform(image)

      age = torch.tensor([sample.age], dtype=torch.float32)
      gender = torch.tensor(self.gender_dict[sample.gender], dtype=torch.int32)
      ethnicity = torch.tensor(
          self.ethnicity_dict[sample.ethnicity], dtype=torch.int32)

      return image, age, gender, ethnicity
