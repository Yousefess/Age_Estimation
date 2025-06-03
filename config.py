import torch

config = {
    "batch_size": 128,
    "batch_size_test": 256,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
