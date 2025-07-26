import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    '''
    Depthwise Separable Convolution Block
    
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1):
        
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.depthwise(x)
        return self.pointwise(out)
    
class SEBlock(nn.Module):
    '''
    SE Block
    
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507v4
    '''
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels//reduction_ratio, in_channels)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.gap(x).squeeze()
        out = self.fc1(out)
        out = self.fc2(self.relu(out))
        out = torch.sigmoid(out)
        return out.view(b, c, 1, 1)
    
                
class BasicBlock(nn.Module):
    '''
    Basic Residual Block
    
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    
    - (conv-batchnorm-relu) x 2 
    - Stride option is given as an argument but not used in this implementation!
    '''
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 apply_SE=True):
        
        super(BasicBlock, self).__init__()
        self.identity = stride == 1
        self.apply_SE = apply_SE
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.SE = SEBlock(out_channels)
        if self.identity:
            if in_channels != out_channels:
                self.projection = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.projection = None
            
    def forward(self, x):
        out = self.relu1(self.batchnorm1(self.conv1(x)))
        out = self.relu2(self.batchnorm2(self.conv2(out)))
        if self.apply_SE:
            out = self.SE(out) * out
        if self.identity:
            if self.projection:
                x = self.projection(x)
            return x + out
        else:
            return out


class BottleneckBlock(nn.Module):
    '''
    Bottleneck Residual Block
    
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385

    - (conv1-batchnorm1-relu1) -> projection
    - (conv2-batchnorm2-relu2) -> bottleneck
    - (conv3-batchnorm3-relu3) -> expansion
    - Stride option is given as an argument but not used in this implementation!    
    '''
    def __init__(self, 
                 in_channels, 
                 projection_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 apply_SE=True):
        
        super(BottleneckBlock, self).__init__()
        self.identity = stride == 1
        self.apply_SE = apply_SE
        self.conv1 = nn.Conv2d(in_channels, projection_channels, 1, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(projection_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(projection_channels, projection_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(projection_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(projection_channels, out_channels, 1, 1, 0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.SE = SEBlock(out_channels)
        if self.identity:
            if in_channels != out_channels:
                self.projection = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.projection = None
            
    def forward(self, x):
        out = self.relu1(self.batchnorm1(self.conv1(x)))
        out = self.relu2(self.batchnorm2(self.conv2(out)))
        out = self.relu3(self.batchnorm3(self.conv3(out)))
        if self.apply_SE:
            out = self.SE(out) * out
        if self.identity:
            if self.projection:
                x = self.projection(x)
            return x + out
        else:
            return out


class FusedMBConv(nn.Module):
    '''
    EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/abs/2104.00298
    '''
    def __init__(self, 
                 in_channels, 
                 expansion_ratio, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 apply_SE=True):
        
        super(FusedMBConv, self).__init__()
        self.identity = stride == 1
        self.conv1 = nn.Conv2d(in_channels, in_channels*expansion_ratio, kernel_size, stride, padding, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(in_channels*expansion_ratio)
        self.silu1 = nn.SiLU(inplace=True)
        if apply_SE:
            self.SE = SEBlock(in_channels*expansion_ratio)
        else:
            self.SE = None
        self.conv2 = nn.Conv2d(in_channels*expansion_ratio, out_channels, kernel_size=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.silu2 = nn.SiLU(inplace=True)
        if self.identity:
            if in_channels != out_channels:
                self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self.projection = None
        
    def forward(self, x):
        out = self.silu1(self.batchnorm1(self.conv1(x)))
        if self.SE:
            out = self.SE(out) * out
        out =  self.silu2(self.batchnorm2(self.conv2(out)))
        if self.identity:
            if self.projection:
                x = self.projection(x)
            return x + out
        else:
            return out
        
        
class MBConv(nn.Module):
    '''
    MnasNet: Platform-Aware Neural Architecture Search for Mobile
    https://arxiv.org/abs/1807.11626
    '''
    def __init__(self, 
                 in_channels, 
                 expansion_ratio, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 apply_SE=True):
        
        super(MBConv, self).__init__()
        self.identity = stride == 1
        self.conv1 = nn.Conv2d(in_channels, in_channels*expansion_ratio, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(in_channels*expansion_ratio)
        self.silu1 = nn.SiLU(inplace=True)
        self.dsconv = DSConv(in_channels*expansion_ratio, in_channels*expansion_ratio, kernel_size, stride, padding)
        self.batchnorm2 = nn.BatchNorm2d(in_channels*expansion_ratio)
        self.silu2 = nn.SiLU(inplace=True)
        if apply_SE:
            self.SE = SEBlock(in_channels*expansion_ratio)
        else:
            self.SE = None
        self.conv2 = nn.Conv2d(in_channels*expansion_ratio, out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.silu3 = nn.SiLU(inplace=True)
        if self.identity:
            if in_channels != out_channels:
                self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.projection = None
        
    def forward(self, x):
        out = self.silu1(self.batchnorm1(self.conv1(x)))
        out = self.silu2(self.batchnorm2(self.dsconv(out)))
        if self.SE:
            out = self.SE(out) * out
        out = self.silu3(self.batchnorm3(self.conv2(out)))
        if self.identity:
            if self.projection:
                x = self.projection(x)
            return x + out
        else:
            return out