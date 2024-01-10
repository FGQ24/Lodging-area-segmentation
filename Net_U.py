import torch
import torch.nn as nn
import torch.nn.functional as F


class Unetconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unetconv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),  # inplace=True，节省内存开销
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        X = self.conv1(X)
        outputs = self.conv2(X)
        return outputs


# ②上采样层
class upconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upconv, self).__init__()

        self.conv = Unetconv(in_channels, out_channels)
        # ①反卷积
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # ②skip connection，数据合并

    def forward(self, inputs_R, inputs_U):
        # self,x2,x1
        outputs_U = self.upconv1(inputs_U)
        offset = outputs_U.size()[-1] - inputs_R.size()[-1]
        pad = [offset // 2, offset - offset // 2, offset // 2, offset - offset // 2]  # 2*[1,1]=[1,1,1,1]
        outputs_R = F.pad(inputs_R, pad)

        # 这里教程写的dim=1，但torch(c,h,w)，我觉得dim=0的时候才是通道相加
        # tensor是四维的，所以dim=1，即按三维拼接
        return self.conv(torch.cat((outputs_U, outputs_R), dim=1))


class U_net(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(U_net, self).__init__()
        self.in_channels = in_channels

        filters = [64, 128, 256, 512, 1024]

        # 下采样
        self.conv1 = Unetconv(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = Unetconv(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = Unetconv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Unetconv(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = Unetconv(filters[3], filters[4])
        # 上采样
        self.upnet4 = upconv(filters[4], filters[3])
        self.upnet3 = upconv(filters[3], filters[2])
        self.upnet2 = upconv(filters[2], filters[1])
        self.upnet1 = upconv(filters[1], filters[0])
        #
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, kernel_size=1),
            nn.Upsample(size=(512, 512)),
        )

    def forward(self, inputs):
        # 下
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        downputs = self.maxpool4(conv4)

        centerputs = self.center(downputs)
        # 上
        up4 = self.upnet4(conv4, centerputs)
        up3 = self.upnet3(conv3, up4)
        up2 = self.upnet2(conv2, up3)
        up1 = self.upnet1(conv1, up2)
        # 1×1
        final = self.final(up1)

        return torch.sigmoid(final)
