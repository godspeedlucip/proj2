import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba import *

class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='same'):
        super(Conv2D, self).__init__()
        padding = kernel_size // 2 if padding == 'same' else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# class DenseBlockMamba(nn.Module):
#     def __init__(self, input_channels, num_filters):
#         super(DenseBlockMamba, self).__init__()
#         self.conv1 = Conv2D(input_channels, num_filters, kernel_size=3)
#         self.mamba = VSSBlock(num_filters)
#         # self.conv1 = Conv2D(input_channels, num_filters, kernel_size=3)
#         # self.conv2 = Conv2D(num_filters, num_filters, kernel_size=3)
#         self.leff = Conv2D(num_filters, num_filters, kernel_size=3)
#         self.skip_conv = nn.Conv2d(input_channels, num_filters, kernel_size=1) if input_channels != num_filters else None

#     def forward(self, x):
#         skip = x if self.skip_conv is None else self.skip_conv(x)
#         out1 = self.conv1(x)
#         out1 = out1.permute(0, 2, 3, 1).contiguous()
#         out2 = self.mamba(out1)
#         out2 = out2.permute(0, 3, 1, 2).contiguous()
#         out3 = self.leff(out2)
#         # out1 = self.conv1(x)
#         # out2 = self.conv2(out1)
#         # out3 = self.conv3(torch.cat([out2, out1], dim=1))
#         return out3 + skip


class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(DenseBlock, self).__init__()
        self.conv1 = Conv2D(input_channels, num_filters, kernel_size=3)
        self.conv2 = Conv2D(num_filters, num_filters, kernel_size=3)
        self.conv3 = Conv2D(num_filters+num_filters, num_filters, kernel_size=3)
        self.skip_conv = nn.Conv2d(input_channels, num_filters, kernel_size=1) if input_channels != num_filters else None

    def forward(self, x):
        skip = x if self.skip_conv is None else self.skip_conv(x)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(torch.cat([out2, out1], dim=1))
        return out3 + skip

class ResBlock(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(input_channels, num_filters, kernel_size=3)
        self.conv2 = Conv2D(num_filters, num_filters, kernel_size=3)
        self.skip_conv = nn.Conv2d(input_channels, num_filters, kernel_size=1) if input_channels != num_filters else None

    def forward(self, x):
        skip = x if self.skip_conv is None else self.skip_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + skip

class Dense2DPrior(nn.Module):
    def __init__(self):
        super(Dense2DPrior,self).__init__()
        self.block1 = DenseBlock(1, 24) # Assuming input channels is 1
        self.block2 = DenseBlock(24, 48)
        self.block3 = DenseBlock(48, 96)
        self.block4 = DenseBlock(96, 96)
        self.block5 = DenseBlock(96, 192)

    def forward(self, x):
        out1 = self.block1(x)
        maxpool_1 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = self.block2(maxpool_1)

        maxpool_2 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = self.block3(maxpool_2)

        maxpool_3 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = self.block4(maxpool_3)

        maxpool_4 = F.max_pool2d(out4, kernel_size=2, stride=2)
        out8 = self.block5(maxpool_4)

        return out1, out2, out3, out4, out8

class Dense2D(nn.Module):
    def __init__(self):
        super(Dense2D,self).__init__()
        self.prior = Dense2DPrior()
        self.block1 = DenseBlock(1, 24) # Assuming input channels is 1
        self.block2 = DenseBlock(48, 48) # 这里需要检查
        self.block3 = DenseBlock(96, 96)
        self.block4 = DenseBlock(192, 96)
        self.block5 = DenseBlock(192, 192)

        self.upsample_conv1 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)
        self.upsample_conv2 = nn.ConvTranspose2d(96, 192, kernel_size=4, stride=2, padding=1)
        self.upsample_conv3 = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1)
        self.upsample_conv4 = nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1)
        # self.upsample_conv5 = nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=1)

        self.block9 = DenseBlock(96, 24) # 这里需要检查
        self.block8 = DenseBlock(192, 48)
        self.block7 = DenseBlock(384, 96)
        self.block6 = DenseBlock(384, 96)
        self.final_conv = nn.Conv2d(24, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x, prior):
        f1, f2, f3, f4, f5 = self.prior(prior)
        # print(f'{f1.shape}, {f2.shape}, {f3.shape}, {f4.shape}, {f5.shape}')

        out1 = self.block1(x)
        # print(f'{out1.shape}')
        out1 = torch.cat([f1, out1], dim=1)


        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)

        out2 = self.block2(out2)
        # print(f'{out2.shape}')
        out2 = torch.cat([f2, out2], dim=1)

        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = self.block3(out3)
        # print(f'{out3.shape}')
        out3 = torch.cat([f3, out3], dim=1)

        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        # print(f'{out4.shape}')
        out4 = self.block4(out4)
        out4 = torch.cat([f4, out4], dim=1)

        out5 = F.max_pool2d(out4, kernel_size=2, stride=2)
        # print(f'{out5.shape}')
        out5 = self.block5(out5)
        out5 = torch.cat([f5, out5], dim=1)

        # print(f'{out1.shape}, {out2.shape}, {out3.shape}, {out4.shape}, {out5.shape}')
        up1 = self.upsample_conv1(out5)
        # print(f'out4:{out4.shape}, up1{up1.shape}')
        up1 = torch.cat([out4, up1], dim=1)
        out6 = self.block6(up1)
        # print(f'out6:{out6.shape}')
        up2 = self.upsample_conv2(out6)
        # print(f'out3 {out3.shape} up2 {up2.shape}')
        up2 = torch.cat([out3, up2], dim=1)

        out7 = self.block7(up2)

        up3 = self.upsample_conv3(out7)
        up3 = torch.cat([out2, up3], dim=1)
        out8 = self.block8(up3)

        up4 = self.upsample_conv4(out8)
        up4 = torch.cat([out1, up4], dim=1)
        out9 = self.block9(up4)
        # print(f'{up1.shape}, {up2.shape}, {up3.shape}, {up4.shape}')
        result = self.final_conv(out9)

        return out1, F.relu(result+x)

if __name__ == '__main__':
    model = Dense2D()
    input = torch.randn(1,1,512, 512)
    prior = torch.randn(1,1,512, 512)
    _1, output = model(input, prior)
    print(output.shape)
    print(_1.shape)
