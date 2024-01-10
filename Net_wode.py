import torch
import torch.nn as nn
import torch.nn.functional as F
from goodthing import NONLocalBlock, ConvolutionLayer, DownSample, upsample_like, channel_attention, \
    CrissCrossAttention, cbam
from model.utils.dca import DCA


# 改动1 # CrissCrossAttention
# 改动2 # ReLU → LeakyReLU
# 改动3 # DCA

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.ReLU_s1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):
        hx = x
        xout = self.ReLU_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    # src = F.upsample(src, size=tar.shape[2:], mode='bilinear') # old version torch
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,
                 attention=True, n=1,
                 input_size=(512, 512),
                 patch_size=32,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1, 1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4, 4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[16, 16, 16, 16, 16, 16],
                           strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8, patch_size // 16,
                                    patch_size // 32],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx11 = self.pool1(hx1)

        hx2 = self.rebnconv2(hx11)
        hx21 = self.pool2(hx2)

        hx3 = self.rebnconv3(hx21)
        hx31 = self.pool3(hx3)

        hx4 = self.rebnconv4(hx31)
        hx41 = self.pool4(hx4)

        hx5 = self.rebnconv5(hx41)
        hx51 = self.pool5(hx5)

        hx6 = self.rebnconv6(hx51)

        if self.attention:
            hx1, hx2, hx3, hx4, hx5, hx6 = self.DCA([hx1, hx2, hx3, hx4, hx5, hx6])

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,
                 attention=True, n=1,
                 input_size=(256, 256),
                 patch_size=16,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[16, 16, 16, 16, 16],
                           strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8, patch_size // 16],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        if self.attention:
            hx1, hx2, hx3, hx4, hx5 = self.DCA([hx1, hx2, hx3, hx4, hx5])

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,
                 attention=True, n=1,
                 input_size=(128, 128),
                 patch_size=8,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[16, 16, 16, 16],
                           strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        if self.attention:
            hx1, hx2, hx3, hx4 = self.DCA([hx1, hx2, hx3, hx4])

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,
                 attention=True, n=1,
                 input_size=(64, 64),
                 patch_size=4,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[16, 16, 16],
                           strides=[patch_size, patch_size // 2, patch_size // 4],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        if self.attention:
            hx1, hx2, hx3 = self.DCA([hx1, hx2, hx3])

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,
                 attention=True, n=1,
                 input_size=(32, 32),
                 patch_size=2,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        if self.attention:
            self.DCA = DCA(n=n,
                           features=[16, 16, 16],
                           strides=[patch_size, patch_size, patch_size],
                           patch=patch,
                           spatial_att=spatial_att,
                           channel_att=channel_att,
                           spatial_head=spatial_head_dim,
                           channel_head=channel_head_dim,
                           )

        self.rebnconv4 = CrissCrossAttention(mid_ch)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        if self.attention:
            hx1, hx2, hx3 = self.DCA([hx1, hx2, hx3])

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


### U^2-Net small ###
class wode(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,
                 attention=True, n=1,
                 input_size=(512, 512),
                 patch_size=16,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=None,
                 channel_head_dim=None,
                 ) -> None:
        super().__init__()
        if channel_head_dim is None:
            channel_head_dim = [1, 1, 1, 1, 1]
        if spatial_head_dim is None:
            spatial_head_dim = [4, 4, 4, 4, 4]

        self.attention = attention
        patch = input_size[0] // patch_size

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = CrissCrossAttention(64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)