import torch
from torch.nn import Module
import torch.nn as nn
import copy

class newPad2d(Module):
    def __init__(self,length):
        super(newPad2d,self).__init__()
        self.length = length
        self.zeroPad = nn.ZeroPad2d(self.length)

    def forward(self, input):
        b,c,h,w = input.shape
        output = self.zeroPad(input)

        #output = torch.FloatTensor(b,c,h+self.length*2,w+self.length*2)
        #output[:,:,self.length:self.length+h,self.length:self.length+w] = input

        for i in range(self.length):
        # 一层的四个切片
            output[:, :, self.length:self.length+h, i] = output[:, :, self.length:self.length+h, self.length]
            output[:, :, self.length:self.length + h, w+ self.length+i] = output[:, :, self.length:self.length + h,
                                                                self.length-1+w]
            output[:, :, i, self.length:self.length+w] = output[:, :, self.length, self.length:self.length+w]
            output[:, :, h+self.length+i, self.length:self.length + w] = output[:, :, h + self.length-1,
                                                                self.length:self.length + w]
         # 对角进行特别处理
        for j in range(self.length):
            for k in range(self.length):
                output[:,:,j,k]=output[:,:,self.length,self.length]
                output[:, :, j, w+ self.length+k] = output[:, :, self.length, self.length-1+w]
                output[:, :, h+self.length+j, k] = output[:, :, h + self.length-1, self.length]
                output[:, :, h+self.length+j, w + self.length + k] = output[:, :, h + self.length-1, self.length - 1 + w]
        return output
