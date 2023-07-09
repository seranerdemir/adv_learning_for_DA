

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
class DeepLabV2(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV2, self).__init__()

        # Pretrained ResNet101
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify stride and dilation
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # ASPP Module
        self.aspp1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.aspp2 = nn.Conv2d(2048, num_classes, kernel_size=3, dilation=6, padding=6)
        self.aspp3 = nn.Conv2d(2048, num_classes, kernel_size=3, dilation=12, padding=12)
        self.aspp4 = nn.Conv2d(2048, num_classes, kernel_size=3, dilation=18, padding=18)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, num_classes, kernel_size=1))

    def forward(self, x):
        h = checkpoint(self.layer0, x) #checkpoint use the memory usage
        h = checkpoint(self.layer1, h)
        h = checkpoint(self.layer2, h)
        h = checkpoint(self.layer3, h)
        h = checkpoint(self.layer4, h)

        h1 = self.aspp1(h)
        h2 = self.aspp2(h)
        h3 = self.aspp3(h)
        h4 = self.aspp4(h)
        h5 = self.global_avg_pool(h)
        h5 = nn.Upsample((h.shape[2], h.shape[3]), mode='bilinear', align_corners=True)(h5)

        output = h1 + h2 + h3 + h4 + h5

        # Upsample to the original image size
        output = F.interpolate(output, size=(512, 1024), mode='bilinear', align_corners=True)


        return output