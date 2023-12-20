import torch
from torch import nn


def ConvBlock(first_chanels, second_chanels): # (batch, 512, ...)
    return nn.Sequential(
            nn.Conv2d(second_chanels, first_chanels, 1), # 512, 256
            nn.BatchNorm2d(first_chanels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(first_chanels, second_chanels, 3, padding=1), # 256, 512
            nn.BatchNorm2d(second_chanels),
            nn.LeakyReLU(0.1, inplace=True),
        )

class Model_CV_Big(nn.Module):
    def __init__(self, pretrain_mode, num_classes):
        super(Model_CV_Big, self).__init___()
        self.pretrain_mode = pretrain_mode
        self.num_classes = num_classes
        
        self.seqns = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), # (batch, 64, 256, 256)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2), # (batch, 64, 128, 128)

            nn.Conv2d(64, 192, 3, padding=1), # (batch, 192, 128, 128)
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2), # (batch, 192, 64, 64)

            nn.Conv2d(192, 128, 1), # (batch, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), # (batch, 256, 64, 64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1), # (batch, 256, 64, 64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1), # (batch, 512, 64, 64)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2), # (batch, 512, 32, 32)

            ConvBlock(256, 512), 
            ConvBlock(256, 512),
            ConvBlock(256, 512),
            ConvBlock(256, 512), # (batch, 512, 32, 32)
            nn.Conv2d(512, 512, 1), # (batch, 512, 32, 32)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1), # (batch, 1024, 32, 32)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2), # (batch, 1024, 16, 16)

            ConvBlock(512, 1024), 
            ConvBlock(512, 1024), # (batch, 1024, 16, 16)
            nn.Conv2d(1024, 1024, 3, padding=1), # (batch, 1024, 16, 16)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1), # (batch, 1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1), # (batch, 1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), # (batch, 1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        if self.pretrain_mode:
            self.fcl = nn.Linear(1024, num_classes)

    def forward(self, x):
        output = self.net(x) # (batch, 1024, 8, 8)

        if self.pretrain_mode:
            output = nn.functional.avg_pool2d(output, (output.size(2), output.size(3))) # (batch, 1024, 1, 1)
            output = output.squeeze(output) # (batch, 1024)
            output = self.fcl(output)

        return output  