'''Defines the architectures of the WGAN Critic and Generator.'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample(x):
    '''Changes a [N, Depth, Height, Width] tensor to [N, Depth, 2 * Height, 2 * Width].'''

    return F.interpolate(x, scale_factor=2, mode='nearest')


class Critic(nn.Module):
    '''Takes a 3x128x128 image and returns a big number if the image is real, or a small number if the image is generated.'''

    def __init__(self):
        super(Critic, self).__init__()

        # Input is 3x128x128, output is 32x64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 32x64x64, output is 64x32x32
        self.conv2_layernorm = nn.LayerNorm([32, 64, 64])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 64x32x32, output is 128x16x16
        self.conv3_layernorm = nn.LayerNorm([64, 32, 32])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 128x16x16, output is 256x8x8
        self.conv4_layernorm = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 256x8x8, output is 512x4x4
        self.conv5_layernorm = nn.LayerNorm([256, 8, 8])
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)

        # Input is 512*4*4, output is 1
        self.fc_layernorm = nn.LayerNorm([512 * 4 * 4])
        self.fc = nn.Linear(512 * 4 * 4, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.conv2_layernorm(x)))
        x = F.relu(self.conv3(self.conv3_layernorm(x)))
        x = F.relu(self.conv4(self.conv4_layernorm(x)))
        x = F.relu(self.conv5(self.conv5_layernorm(x))).view(-1, 512 * 4 * 4)
        x = self.fc(self.fc_layernorm(x))

        return x

class Generator(nn.Module):
    '''Takes a 512-dimensional latent space vector and generates a 3x128x128 image.'''

    def __init__(self):
        super(Generator, self).__init__()
        # Input is a latent space vector of size 512, output is 512*4*4
        self.fc = nn.Linear(512, 512 * 4 * 4)

        # Input is 512x8x8, output is 256x8x8
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 256x16x16, output is 128x16x16
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 128x32x32, output is 64x32x32
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 64x64x64, output is 32x64x64
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        # Input is 32x128x128, output is 3x128x128
        self.conv5_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x)).view(-1, 512, 4, 4)

        x = _upsample(x)
        x = F.relu(self.conv1(self.conv1_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv2(self.conv2_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv3(self.conv3_bn(x)))

        x = _upsample(x)
        x = F.relu(self.conv4(self.conv4_bn(x)))

        x = _upsample(x)
        x = torch.tanh(self.conv5(self.conv5_bn(x)))

        return x
