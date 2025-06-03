import os
import pandas as pd

# Path to the UTKFace dataset directory
dataset_dir = 'data/UTKFace'

# Initialize lists to store data
image_names = []
ages = []
ethnicities = []
genders = []

# Iterate through files in the directory
for filename in os.listdir(dataset_dir):
  if filename.endswith('.jpg'):
    parts = filename.split('_')
    if len(parts) < 4:
      print(filename)
      continue

    age = int(parts[0])
    gender = 'Male' if int(parts[1]) == 0 else 'Female'
    ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(parts[2])]

    if age > 80:
      continue

    image_names.append(filename)
    ages.append(age)
    genders.append(gender)
    ethnicities.append(ethnicity)

# Create a DataFrame the lists
data = {
    'image_name': image_names,
    'age': ages,
    'ethnicity': ethnicities,
    'gender': genders
}
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_filename = 'data/utkface_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully.")
print(f"Length of dataset: {len(df)}")
