import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/utkface_dataset.csv')

df_train, temp = train_test_split(
    df, test_size=0.3, stratify=df.age, random_state=42)
df_test, df_valid = train_test_split(
    temp, test_size=0.5, stratify=temp.age, random_state=42)

print(f"Shape of train: {df_train.shape},\nShape of test: {df_test.shape},\nShape of validation: {df_valid.shape}")

df_train.to_csv('data/train_set.csv', index=False)
df_valid.to_csv('data/valid_set.csv', index=False)
df_test.to_csv('data/test_set.csv', index=False)

print('All CSV files created successfully.')
