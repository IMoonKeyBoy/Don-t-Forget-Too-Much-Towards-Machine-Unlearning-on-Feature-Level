import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import BasicBlock, BottleneckBlock, MBConv, FusedMBConv


class CustomResNetFor_Single(nn.Module):
    '''
    Assume |C x H x W| : |3 x 224 x 224|

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    '''

    def __init__(self):
        super(CustomResNetFor_Single, self).__init__()
        self.block1 = BasicBlock(3, 32, 64)
        self.pool1 = nn.MaxPool2d((2, 2))  # |out| = |64 x 112 x 112|
        self.block2 = BottleneckBlock(64, 32, 128)
        self.block2_2 = BottleneckBlock(128, 32, 128)
        self.pool2 = nn.MaxPool2d((2, 2))  # |out| = |128 x 56 x 56|
        self.block3 = BottleneckBlock(128, 32, 256)
        self.block3_2 = BottleneckBlock(256, 32, 256)
        self.pool3 = nn.MaxPool2d((2, 2))  # |out| = |256 x 28 x 28|
        self.block4 = BottleneckBlock(256, 64, 512)
        self.block4_2 = BottleneckBlock(512, 64, 512)
        self.pool4 = nn.MaxPool2d((2, 2))  # |out| = |512 x 14 x 14|
        self.block5 = BottleneckBlock(512, 64, 512)
        self.block5_2 = BottleneckBlock(512, 64, 512)
        self.pool5 = nn.MaxPool2d((2, 2))  # |out| = |512 x 7 x 7|
        self.block6 = BottleneckBlock(512, 64, 512)
        self.block6_2 = BottleneckBlock(512, 64, 1024, apply_SE=False)  # |out| = |512 x 7 x 7|
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        out = self.pool1(self.block1(x))
        out = self.pool2(self.block2_2(self.block2(out)))
        out = self.pool3(self.block3_2(self.block3(out)))
        out = self.pool4(self.block4_2(self.block4(out)))
        out = self.pool5(self.block5_2(self.block5(out)))
        out = self.block6_2(self.block6(out))
        out = self.gap(out).squeeze()
        out = self.linear2(self.linear1(out))
        return out


class CustomResNetFor_Multi(nn.Module):
    '''
    Assume |C x H x W| : |3 x 224 x 224|

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    '''

    def __init__(self, out_number):
        super(CustomResNetFor_Multi, self).__init__()
        self.block1 = BasicBlock(3, 32, 64)
        self.pool1 = nn.MaxPool2d((2, 2))  # |out| = |64 x 112 x 112|
        self.block2 = BottleneckBlock(64, 32, 128)
        self.block2_2 = BottleneckBlock(128, 32, 128)
        self.pool2 = nn.MaxPool2d((2, 2))  # |out| = |128 x 56 x 56|
        self.block3 = BottleneckBlock(128, 32, 256)
        self.block3_2 = BottleneckBlock(256, 32, 256)
        self.pool3 = nn.MaxPool2d((2, 2))  # |out| = |256 x 28 x 28|
        self.block4 = BottleneckBlock(256, 64, 512)
        self.block4_2 = BottleneckBlock(512, 64, 512)
        self.pool4 = nn.MaxPool2d((2, 2))  # |out| = |512 x 14 x 14|
        self.block5 = BottleneckBlock(512, 64, 512)
        self.block5_2 = BottleneckBlock(512, 64, 512)
        self.pool5 = nn.MaxPool2d((2, 2))  # |out| = |512 x 7 x 7|
        self.block6 = BottleneckBlock(512, 64, 512)
        self.block6_2 = BottleneckBlock(512, 64, 1024, apply_SE=False)  # |out| = |512 x 7 x 7|
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, out_number)

    def forward(self, x):
        out = self.pool1(self.block1(x))
        out = self.pool2(self.block2_2(self.block2(out)))
        out = self.pool3(self.block3_2(self.block3(out)))
        out = self.pool4(self.block4_2(self.block4(out)))
        out = self.pool5(self.block5_2(self.block5(out)))
        out = self.block6_2(self.block6(out))
        out = self.gap(out).squeeze()
        return F.sigmoid(self.linear2(self.linear1(out)))


class CustomEfficientNet(nn.Module):
    '''
    Customized version of EfficientNetV2
    
    EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/abs/2104.00298
    '''

    def __init__(self) -> None:
        super(CustomEfficientNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.fusedmbconv1 = FusedMBConv(24, 1, 24, 3)
        self.fusedmbconv2 = FusedMBConv(24, 4, 48, 3)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fusedmbconv3 = FusedMBConv(48, 4, 64, 3)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.mbconv1 = MBConv(64, 4, 128, 3)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.mbconv2 = MBConv(128, 6, 160, 3)
        self.mbconv3 = MBConv(160, 6, 256, 3, apply_SE=False)
        self.pool5 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(256, 1280, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(1280, 256)
        self.linear2 = nn.Linear(256, 3)

    def forward(self, x):
        out = self.pool1(self.conv1(x))
        out = self.fusedmbconv1(out)
        out = self.pool2(self.fusedmbconv2(out))
        out = self.pool3(self.fusedmbconv3(out))
        out = self.pool4(self.mbconv1(out))
        out = self.mbconv2(out)
        out = self.pool5(self.mbconv3(out))
        out = self.conv2(out)
        out = self.gap(out).squeeze()
        out = self.linear2(self.linear1(out))
        return torch.sigmoid(out)


class EfficeintNetV2(nn.Module):
    '''
    EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/abs/2104.00298
    
    Highly influenced by
    https://github.com/d-li14/efficientnetv2.pytorch
    '''

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        input_channel = 24
        layers = [BasicBlock(3, input_channel // 2, input_channel, stride=2, padding=1)]
        for e, o, n, s, se, b in self.model_cfg:
            block = b
            for i in range(n):
                layers.append(block(input_channel, e, o,
                                    stride=s if i == 0 else 1,
                                    apply_SE=se))
                input_channel = o
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(p=0.5)
        self.conv = nn.Conv2d(input_channel, 1280, 1, bias=False)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(1280, 3)

        self._initialize_weights()

    def forward(self, x):
        out = self.layers(x)
        out = self.drop(out)  # added
        out = self.conv(out)
        out = self.pool(out).squeeze()
        return torch.sigmoid(self.classifier(out))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s():
    model_cfg = [
        # expansion_ratio, out_channel, num, stride, apply_se
        [1, 24, 2, 1, True, FusedMBConv],  # Fused_MBConv
        [4, 48, 4, 2, True, FusedMBConv],  # Fused_MBConv
        [4, 64, 4, 2, True, FusedMBConv],  # Fused_MBConv
        [4, 128, 6, 2, True, MBConv],  # MBConv
        [6, 160, 9, 1, True, MBConv],  # MBConv
        [6, 256, 15, 2, True, MBConv]  # MBConv
    ]
    return EfficeintNetV2(model_cfg)


def effnetv2_m():
    # To be updated
    pass
