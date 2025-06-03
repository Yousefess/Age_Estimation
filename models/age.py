import torch.nn as nn
import torchvision.models as models

class AgeEstimationModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    self.model.fc = nn.Linear(in_features=2048, out_features=1)

  def forward(self, x):
    y = self.model(x)
    return y
