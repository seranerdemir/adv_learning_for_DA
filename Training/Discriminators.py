import torch
import torch.nn as nn

class PixelwiseDiscriminator(nn.Module):
    def __init__(self, num_classes=19):
        super(PixelwiseDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.out_channels = 64
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(self.num_classes, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels*2, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        output = self.softmax(input)
        output = self.conv1(output)
        output = self.leaky_relu1(output)

        output = self.conv2(output)
        output = self.leaky_relu2(output)

        output = self.conv3(output)

        return output


class ImagewiseDiscriminator(nn.Module):
    def __init__(self, num_classes=19):
        super(ImagewiseDiscriminator, self).__init__()

        self.num_classes = num_classes
        self.out_channels = 64
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=self.num_classes, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=4, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels*2, out_channels=self.out_channels*4, kernel_size=4, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=self.out_channels*4, out_channels=self.out_channels*8, kernel_size=4, stride=2, padding=1)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=self.out_channels*8, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, img):
        output = self.softmax(img)
        output = self.conv1(output)
        output = self.leaky_relu1(output)

        output = self.conv2(output)
        output = self.leaky_relu2(output)

        output = self.conv3(output)
        output = self.leaky_relu3(output)

        output = self.conv4(output)
        output = self.leaky_relu4(output)

        output = self.conv5(output)

        return output