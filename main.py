from data import UTKDataset
from models import AgeEstimationModel
from config import config
from train import train_one_epoch, evaluate

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchmetrics as tm

# =============== Transformation ===============
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============== Create Datasets ===============
dataset_dir = 'data/UTKFace'

train_set = UTKDataset(
    root_dir=dataset_dir, csv_file='data/train_set.csv', transform=train_transform
)

valid_set = UTKDataset(
    root_dir=dataset_dir, csv_file='data/valid_set.csv', transform=test_transform
)

test_set = UTKDataset(
    root_dir=dataset_dir, csv_file='data/test_set.csv', transform=test_transform
)

# =============== Define DataLoaders ===============
train_loader = DataLoader(
    train_set, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(
    valid_set, batch_size=config['batch_size_test'], shuffle=False)
test_loader = DataLoader(
    test_set, batch_size=config['batch_size_test'], shuffle=False)

# =============== Define Model ===============
model = AgeEstimationModel().to(config['device'])

loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                      momentum=0.9, weight_decay=config['weight_decay'])
metric = tm.MeanAbsoluteError().to(config['device'])

# =============== Train Model ===============
loss_train_hist = []
loss_valid_hist = []

metric_train_hist = []
metric_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

num_epochs = 10

for epoch in range(num_epochs):
    # Train
    model, loss_train, metric_train = train_one_epoch(model,
                                                      train_loader,
                                                      loss_fn,
                                                      optimizer,
                                                      metric,
                                                      epoch)
    # Validation
    loss_valid, metric_valid = evaluate(model,
                                        valid_loader,
                                        loss_fn,
                                        metric)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)

    if loss_valid < best_loss_valid:
        torch.save(model, f'models/best_age_estimation_model.pt')
        best_loss_valid = loss_valid
        print('Model Saved!')

    print(f'Valid: Loss = {loss_valid:.4}, MAE = {metric_valid:.4}')
    print()

    epoch_counter += 1

# =============== Plot Loss ===============
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(range(epoch_counter), loss_train_hist, 'r-', label='Train')
ax.plot(range(epoch_counter), loss_valid_hist, 'b-', label='Validation')

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()

fig.savefig('Loss.png', dpi=300)
