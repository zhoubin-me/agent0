import numpy as np
import torch
import torch.optim as optim

from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, train, test
import ray
import sys

ray.init(address="10.80.43.126:6379")

import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_mnist(config):
    model = ConvNet()
    train_loader, test_loader = get_data_loaders()
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader, torch.device("cpu"))
        acc = test(model, test_loader, torch.device("cpu"))
        tune.track.log(mean_accuracy=acc)
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")

from ray.tune.schedulers import ASHAScheduler

search_space = {
    "lr": tune.choice([0.001, 0.01, 0.1]),
    "momentum": tune.uniform(0.1, 0.9)
}

analysis = tune.run(
    train_mnist,
    num_samples=30,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=1),
    config=search_space)
