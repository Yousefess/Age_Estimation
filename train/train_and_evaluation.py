from utils.helpers import AverageMeter
from config import config

import torch
import tqdm


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
  model.train()
  loss_train = AverageMeter()
  metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets, *_ in tepoch:
      inputs, targets = inputs.to(config['device']), targets.to(config['device'])
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)

      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

  return model, loss_train.avg, metric.compute().item()


def evaluate(model, test_loader, loss_fn, metric):
  model.eval()
  loss_eval = AverageMeter()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets, *_ in test_loader:
      inputs, targets = inputs.to(config['device']), targets.to(config['device'])
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)

      loss_eval.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)

  return loss_eval.avg, metric.compute().item()
