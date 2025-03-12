# Some basic layers.
# 一些基础的层。

import torch
import torch.nn as nn
import torch.nn.functional as F


# UNet
class UNetBlock(nn.Module):
    def __init__(self, num_layers=3, in_channels=3, out_channels=3, filters=32):
        super(UNetBlock, self).__init__()

        # TODO: 暂时用SID的四下四上模型替代
        # 原来只是一个Unet（四下四上）
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1_1(x))
        conv1 = F.leaky_relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2_1(pool1))
        conv2 = F.leaky_relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = F.leaky_relu(self.conv3_1(pool2))
        conv3 = F.leaky_relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = F.leaky_relu(self.conv4_1(pool3))
        conv4 = F.leaky_relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        conv5 = F.leaky_relu(self.conv5_1(pool4))
        conv5 = F.leaky_relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = F.leaky_relu(self.conv6_1(up6))
        conv6 = F.leaky_relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = F.leaky_relu(self.conv7_1(up7))
        conv7 = F.leaky_relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = F.leaky_relu(self.conv8_1(up8))
        conv8 = F.leaky_relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = F.leaky_relu(self.conv9_1(up9))
        conv9 = F.leaky_relu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)

        # pixel_shuffle
        out = nn.functional.pixel_shuffle(conv10, 2)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


# ResNet
class ResNetBlock(nn.Module):
    def __init__(self, num_layers=3, in_channels=1, out_channels=1, filters=32):
        super(ResNetBlock, self).__init__()


# VGG
class VGGBlock(nn.Module):
    def __init__(self, structure=None, in_channels=3, out_channels=10):
        """
        VGG

        :param structure: (default)VGG16
        :param in_channels:
        :param out_channels:
        """
        super(VGGBlock, self).__init__()
        if structure is None:
            structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        for x in structure:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.features = nn.Sequential(*layers)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),        # It seems that 0.4 is better than 0.5
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(4096, out_channels)
        # softmax

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out


