import functools

import torch
import torch.nn as nn
import torch.nn.utils
from cprint import cprint
from torch.autograd import Function

from functions import get_single_classification_model


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# with skip connection and pixel connection and smoothed
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = True
        # construct unet structure
        self.downsample_0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downRelu_1 = nn.LeakyReLU(0.2, True)
        self.downSample_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_1 = norm_layer(ngf * 2)

        self.downRelu_2 = nn.LeakyReLU(0.2, True)
        self.downSample_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_2 = norm_layer(ngf * 4)

        self.downRelu_3 = nn.LeakyReLU(0.2, True)
        self.downSample_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_3 = norm_layer(ngf * 8)

        self.innerLeakyRelu = nn.LeakyReLU(0.2, True)
        self.innerDownSample = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.innerRelu = nn.ReLU(True)
        innerUpSample = []
        innerUpSample.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        innerUpSample.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        innerUpSample.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.innerUpSample = nn.Sequential(*innerUpSample)

        self.innerNorm = norm_layer(ngf * 8)

        self.upRelu_3 = nn.ReLU(True)
        upSample_3 = []
        upSample_3.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        upSample_3.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_3.append(nn.Conv2d(ngf * 16, ngf * 4, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_3 = nn.Sequential(*upSample_3)
        self.upNorm_3 = norm_layer(ngf * 4)

        self.upRelu_2 = nn.ReLU(True)
        upSample_2 = []
        upSample_2.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        upSample_2.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_2.append(nn.Conv2d(ngf * 8, ngf * 2, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_2 = nn.Sequential(*upSample_2)
        self.upNorm_2 = norm_layer(ngf * 2)

        self.upRelu_1 = nn.ReLU(True)
        upSample_1 = []
        upSample_1.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        upSample_1.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_1.append(nn.Conv2d(ngf * 4, ngf, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_1 = nn.Sequential(*upSample_1)
        self.upNorm_1 = norm_layer(ngf)

        self.upRelu_0 = nn.ReLU(True)
        upSample_0 = []
        upSample_0.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        upSample_0.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_0.append(nn.Conv2d(ngf * 2, 1, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_0 = nn.Sequential(*upSample_0)

        ## initialize bias
        nn.init.normal_(self.upSample_0[-1].bias, mean=3, std=1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # assume input image size = 224
        x_down_0 = self.downsample_0(input)  # (ngf, 112, 112)

        x_down_1 = self.downNorm_1(self.downSample_1(self.downRelu_1(x_down_0)))  # (ngf*2, 56, 56)
        x_down_2 = self.downNorm_2(self.downSample_2(self.downRelu_2(x_down_1)))  # (ngf*4, 28, 28)
        x_down_3 = self.downNorm_3(self.downSample_3(self.downRelu_3(x_down_2)))  # (ngf*8, 14, 14)

        latent = self.innerDownSample(self.innerLeakyRelu(x_down_3))  # (ngf*8, 7, 7)

        x = self.innerNorm(self.innerUpSample(self.innerRelu(latent)))  # (ngf*8, 14, 14)

        x_up_3 = self.upNorm_3(self.upSample_3(self.upRelu_3(torch.cat([x, x_down_3], 1))))  # (ngf*4, 28, 28)
        x_up_2 = self.upNorm_2(self.upSample_2(self.upRelu_2(torch.cat([x_up_3, x_down_2], 1))))  # (ngf*2, 56, 56)
        x_up_1 = self.upNorm_1(self.upSample_1(self.upRelu_1(torch.cat([x_up_2, x_down_1], 1))))  # (ngf, 112, 112)

        encoded_image = self.activation(
            self.upSample_0(self.upRelu_0(torch.cat([x_up_1, x_down_0], 1))))  # (3, 224, 224)

        return torch.mul(input, encoded_image), latent


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class ObjectMultiLabelAdv(nn.Module):

    def __init__(self, args, adv_lambda):

        super(ObjectMultiLabelAdv, self).__init__()
        self.args = args
        self.adv_lambda = adv_lambda

        # Construct the U-net
        norm_layer = 'batch'
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.encoder_network = UnetGenerator(3, 3, 5, 64, norm_layer=norm_layer, use_dropout=False)

        # Construct the original model
        self.original_network = get_single_classification_model(args.model)
        cprint.info("load ", args.pretrained_task_network_path)
        self.original_network = torch.load(args.pretrained_task_network_path)

        # Construct the Adversary model
        self.adv_component = get_single_classification_model(args.model)
        cprint.info("load ", args.pretrained_target_network_path)
        self.adv_component = torch.load(args.pretrained_target_network_path)

        # Parameters Freeze
        if not args.autoencoder_finetune:
            for param in self.encoder_network.parameters():
                param.requires_grad = False
        else:
            cprint.warn("Update the Autoencoder model")

        if not args.finetune:
            for param in self.original_network.parameters():
                param.requires_grad = False
        else:
            cprint.warn("Update the Task model")

        if not args.finetune:
            for param in self.adv_component.parameters():
                param.requires_grad = False
        else:
            cprint.warn("Update the Gender Classification model")

    def forward(self, image):

        auto_encoded_image, latent = self.encoder_network(image)

        task_pred = self.original_network(auto_encoded_image)

        adv_feature = ReverseLayerF.apply(auto_encoded_image, self.adv_lambda)

        target_pred = self.adv_component(adv_feature)

        return task_pred, target_pred, auto_encoded_image


    def training_autoencoder(self):
        for param in self.encoder_network.parameters():
            param.requires_grad = True

        for param in self.original_network.parameters():
            param.requires_grad = False

        for param in self.adv_component.parameters():
            param.requires_grad = False

    def training_original(self):

        for param in self.encoder_network.parameters():
            param.requires_grad = False

        for param in self.original_network.parameters():
            param.requires_grad = True

        for param in self.adv_component.parameters():
            param.requires_grad = False

    def training_adv_component(self):

        for param in self.encoder_network.parameters():
            param.requires_grad = False

        for param in self.original_network.parameters():
            param.requires_grad = False

        for param in self.adv_component.parameters():
            param.requires_grad = True

