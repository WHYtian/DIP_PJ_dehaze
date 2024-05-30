import torch
import torch.nn.functional as F
from torch import nn
from skimage.color import rgb2lab, lab2rgb
import torchvision.models as models
from resnext import ResNeXt101
from scipy.ndimage import gaussian_filter

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        rgb_mean = (0.485, 0.456, 0.406)
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.229, 0.224, 0.225)
        self.std = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


# 定义了具有固定均值和标准差的张量参数
class BaseA(nn.Module):
    def __init__(self):
        super(BaseA, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63438
        self.mean[0, 1, 0, 0] = 0.59396
        self.mean[0, 2, 0, 0] = 0.58369
        self.std[0, 0, 0, 0] = 0.16195
        self.std[0, 1, 0, 0] = 0.16937
        self.std[0, 2, 0, 0] = 0.17564

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class BaseITS(nn.Module):
    def __init__(self):
        super(BaseITS, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63542
        self.mean[0, 1, 0, 0] = 0.59579
        self.mean[0, 2, 0, 0] = 0.58550
        self.std[0, 0, 0, 0] = 0.14470
        self.std[0, 1, 0, 0] = 0.14850
        self.std[0, 2, 0, 0] = 0.15348

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class Base_OHAZE(nn.Module):
    def __init__(self):
        super(Base_OHAZE, self).__init__()
        rgb_mean = (0.47421, 0.50878, 0.56789)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.10168, 0.10488, 0.11524)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)

        rgb_mean = (0.35851, 0.35316, 0.34425)
        self.mean_out = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.16391, 0.16174, 0.17148)
        self.std_out = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class J0(Base):
    def __init__(self, num_features=128):
        super(J0, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        a = self.a(f)
        t = F.upsample(self.t(f), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0, max=1)

        if self.training:
            return x_phy, t, a.view(x.size(0), -1)
        else:
            return x_phy


class J1(Base):
    def __init__(self, num_features=128):
        super(J1, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))

        p0 = F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        return x_p0


class J2(Base):
    def __init__(self, num_features=128):
        super(J2, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        p1 = F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        return x_p1


class J3(Base):
    def __init__(self, num_features=128):
        super(J3, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        p2 = F.upsample(self.p2(f), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        return x_p2


class J4(Base):
    def __init__(self, num_features=128):
        super(J4, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))

        p3 = F.upsample(self.p3(f), size=x0.size()[2:], mode='bilinear')
        # x_p3 = (torch.log(1 + p3 * x0)).clamp(min=0, max=1)
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        return x_p3


class ours_wo_AFIM(Base):
    def __init__(self, num_features=128):
        super(ours_wo_AFIM, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        a = self.a(f)
        t = F.upsample(self.t(f), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0, max=1)

        p0 = F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        p1 = F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        p2 = F.upsample(self.p2(f), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        p3 = F.upsample(self.p3(f), size=x0.size()[2:], mode='bilinear')
        # x_p3 = (torch.log(1 + p3 * x0)).clamp(min=0, max=1)
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        attention_fusion = F.upsample(self.attentional_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 5, :, :], 1) * torch.stack(
            (x_phy[:, 0, :, :], x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) * torch.stack((x_phy[:, 1, :, :],
                                                                                                      x_p0[:, 1, :, :],
                                                                                                      x_p1[:, 1, :, :],
                                                                                                      x_p2[:, 1, :, :],
                                                                                                      x_p3[:, 1, :, :]),
                                                                                                     1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) * torch.stack((x_phy[:, 2, :, :],
                                                                                                    x_p0[:, 2, :, :],
                                                                                                    x_p1[:, 2, :, :],
                                                                                                    x_p2[:, 2, :, :],
                                                                                                    x_p3[:, 2, :, :]),
                                                                                                   1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_phy, x_p0, x_p1, x_p2, x_p3, t, a.view(x.size(0), -1)
        else:
            return x_fusion


class ours_wo_J0(Base):
    def __init__(self, num_features=128):
        super(ours_wo_J0, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.attention0 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 12, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        n, c, h, w = down1.size()

        attention0 = self.attention0(concat)
        attention0 = F.softmax(attention0.view(n, 4, c, h, w), 1)
        f0 = down1 * attention0[:, 0, :, :, :] + down2 * attention0[:, 1, :, :, :] + \
             down3 * attention0[:, 2, :, :, :] + down4 * attention0[:, 3, :, :, :]
        f0 = self.refine(f0) + f0

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        p0 = F.upsample(self.p0(f0), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        p1 = F.upsample(self.p1(f1), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        p2 = F.upsample(self.p2(f2), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        p3 = F.upsample(self.p3(f3), size=x0.size()[2:], mode='bilinear')
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        attention_fusion = F.upsample(self.attentional_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 4, :, :], 1) * torch.stack(
            (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 4: 8, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
                                                                                                     x_p1[:, 1, :, :],
                                                                                                     x_p2[:, 1, :, :],
                                                                                                     x_p3[:, 1, :, :]),
                                                                                                    1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 8:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
                                                                                                   x_p1[:, 2, :, :],
                                                                                                   x_p2[:, 2, :, :],
                                                                                                   x_p3[:, 2, :, :]),
                                                                                                  1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_p0, x_p1, x_p2, x_p3
        else:
            return x_fusion
        
def get_gaussian_kernel(kernel_size=15, sigma=1.0):
    """生成高斯核"""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] @ gauss[None, :]
    return kernel

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=1.0):
        super(GaussianSmoothing, self).__init__()
        kernel = get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        weight = self.weight.to(x.device)
        return F.conv2d(x, self.weight, groups=self.groups, padding=self.padding)


class LocalContrastEnhancement(nn.Module):
    def __init__(self, kernel_size=15):
        super(LocalContrastEnhancement, self).__init__()
        self.kernel_size = kernel_size
        

    def forward(self, x):
        # 获取通道数
        channels = x.shape[1]
        # 均值滤波器
        mean_filter = torch.ones((channels, 1, self.kernel_size, self.kernel_size), device=x.device) / (self.kernel_size ** 2)
        # 计算局部均值
        local_mean = F.conv2d(x, mean_filter, padding=self.kernel_size // 2, groups=channels)
        # 计算局部方差
        local_var = F.conv2d(x ** 2, mean_filter, padding=self.kernel_size // 2, groups=channels) - local_mean ** 2
        local_var = local_var.clamp(min=1e-6)
        # 计算标准差
        stddev = torch.sqrt(local_var)
        # 增强对比度
        x_enhanced = (x - local_mean) / (stddev + 1e-6)
        #x_enhanced = x_enhanced.clamp(min=0, max=1)  # 保证像素值在 0 到 1 之间
        # smoothing = GaussianSmoothing(3, 15, 1.0).to(x.device)
        # x_enhanced = smoothing(x_enhanced)
        return x_enhanced
def local_contrast_enhancement(x, kernel_size=15):
    # 初始化一个与输入图像大小相同的张量用于存储增强后的图像
    x_enhanced = torch.zeros_like(x)
    # 对每个通道分别进行处理
    for c in range(x.shape[1]):
        channel = x[:, c:c+1, :, :]  # 选择单个通道
        # 创建均值滤波器
        mean_filter = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
        # 计算局部均值
        local_mean = F.conv2d(channel, mean_filter, padding=kernel_size // 2)
        # 计算局部方差
        local_var = F.conv2d(channel ** 2, mean_filter, padding=kernel_size // 2) - local_mean ** 2
        local_var = local_var.clamp(min=1e-6)
        # 计算标准差
        stddev = torch.sqrt(local_var)
        # 增强对比度
        enhanced_channel = (channel - local_mean) / (stddev + 1e-6)
        # 将增强后的通道赋值回结果张量中
        x_enhanced[:, c:c+1, :, :] = enhanced_channel
    # 重新缩放到0-1范围
    x_enhanced = (x_enhanced - x_enhanced.min()) / (x_enhanced.max() - x_enhanced.min())
    
    return x_enhanced*x

class HistogramEqualization(nn.Module):
    def __init__(self):
        super(HistogramEqualization, self).__init__()

    def forward(self, x):
        # 获取通道数
        channels = x.shape[1]
        # 将图片拉成一维向量
        flattened = x.view(-1)
        # 计算直方图
        hist = torch.histc(flattened, bins=256, min=0, max=1)
        # 计算累积分布函数
        cdf = hist.cumsum(0)
        # 归一化到[0,1]
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        # 根据累积分布函数映射像素值
        x_equalized = torch.interp(flattened, torch.arange(256), cdf)
        # 将处理后的数据恢复形状
        x_equalized = x_equalized.view(x.shape)
        return x_equalized
    
def check_tensor_range(tensor, min_val=0.0, max_val=1.0):
    if torch.isnan(tensor).any():
        nan_mask = torch.isnan(tensor.detach())
        nan_indices = torch.nonzero(nan_mask)
        print("预测结果中NaN 值的位置下标:", nan_indices)
    if torch.isinf(tensor).any():
        print("Tensor contains Inf values!")
    tensor = torch.clamp(tensor, min_val, max_val)
    return tensor
    
class DM2FNet(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d'):
        super(DM2FNet, self).__init__()
        self.num_features = num_features

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=True)
        del backbone.fc
        self.backbone = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention5 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        #self.j5 = LocalContrastEnhancement(kernel_size=15)
        #self.contrast_enhancer = LocalContrastEnhancement(kernel_size=15)
        self.j5 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 18, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):
        x = (x0 - self.mean) / self.std

        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        n, c, h, w = down1.size()

        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 4, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :]
        f_phy = self.refine(f_phy) + f_phy

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
             down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4

        attention5 = self.attention5(concat)
        attention5 = F.softmax(attention5.view(n, 4, c, h, w), 1)
        f5 = down1 * attention5[:, 0, :, :, :] + down2 * attention5[:, 1, :, :, :] + \
             down3 * attention5[:, 2, :, :, :] + down4 * attention5[:, 3, :, :, :]
        f5 = self.refine(f5) + f5

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(f_phy)
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)

        # J2 = I * exp(R2)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)

        #
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)

        r5 = F.upsample(self.j5(f5), size=x0.size()[2:], mode='bilinear')
        x_enhacer = self.contrast_enhancer(x0)
        #x_enhacer = local_contrast_enhancement(r5)
        x_j5 = ((x + x_enhacer*r5) * self.std + self.mean).clamp(min=0., max=1.)
        #x_j5 = (r5+x0+x_enhacer*0.1).clamp(min=0., max=1.)
        #x_j5 = (r5+x0)

        attention_fusion = F.upsample(self.attention_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :6, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                              x_j3[:, 0, :, :], x_j4[:, 0, :, :], x_j5[:, 0, :, :]), 1), 1, True)
        x_f1 = torch.sum(F.softmax(attention_fusion[:, 6: 12, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                              x_j3[:, 1, :, :], x_j4[:, 1, :, :], x_j5[:, 1, :, :]), 1), 1, True)
        x_f2 = torch.sum(F.softmax(attention_fusion[:, 12:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                              x_j3[:, 2, :, :], x_j4[:, 2, :, :], x_j5[:, 2, :, :]), 1), 1, True)
        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)

        # if self.training:
        #     return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, x_j5, t, a.view(x.size(0), -1)
        # else:
        #     return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, x_j5
        return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, x_j5, t, r1, r2, r3, r4, r5
        #return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, x_j5, t



class DM2FNet_woPhy(Base_OHAZE):
    def __init__(self, num_features=64, arch='resnext101_32x8d'):
        super(DM2FNet_woPhy, self).__init__()
        self.num_features = num_features

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=True)
        del backbone.fc
        self.backbone = backbone

        self.down0 = nn.Sequential(
            nn.Conv2d(64, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.fuse3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse0 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

        self.fuse3_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse2_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse1_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse0_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        #self.contrast_enhancer = LocalContrastEnhancement(kernel_size=15)

        self.p4_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        # self.p4_1 = nn.Sequential(
        #     nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features // 2, 3, kernel_size=1)
        # )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=3, padding=1)
        )

        # self.vgg = VGGF()

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean_in) / self.std_in

        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        down0 = self.down0(layer0)
        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        fuse3_attention = self.fuse3_attention(torch.cat((down4, down3), 1))
        f = down4 + self.fuse3(torch.cat((down4, fuse3_attention * down3), 1))

        f = F.upsample(f, size=down2.size()[2:], mode='bilinear')
        fuse2_attention = self.fuse2_attention(torch.cat((f, down2), 1))
        f = f + self.fuse2(torch.cat((f, fuse2_attention * down2), 1))

        f = F.upsample(f, size=down1.size()[2:], mode='bilinear')
        fuse1_attention = self.fuse1_attention(torch.cat((f, down1), 1))
        f = f + self.fuse1(torch.cat((f, fuse1_attention * down1), 1))

        f = F.upsample(f, size=down0.size()[2:], mode='bilinear')
        fuse0_attention = self.fuse0_attention(torch.cat((f, down0), 1))
        f = f + self.fuse0(torch.cat((f, fuse0_attention * down0), 1))

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        r0 = F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + r0).clamp(min=0, max=1)

        r1 = F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + r1) * self.std_out + self.mean_out)\
            .clamp(min=0., max=1.)

        r2_0 = F.upsample(self.p2_0(f), size=x0.size()[2:], mode='bilinear')
        log_x_p2_0 = torch.log(
            ((x + r2_0) * self.std_out + self.mean_out)
                .clamp(min=1e-8))
        r2_1 = F.upsample(self.p2_1(f), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(log_x_p2_0 + r2_1)\
            .clamp(min=0., max=1.)

        r3_0 = F.upsample(self.p3_0(f), size=x0.size()[2:], mode='bilinear')
        log_x_p3_0 = torch.exp(log_log_x0_inverse + r3_0)
        r3_1 = F.upsample(self.p3_1(f), size=x0.size()[2:], mode='bilinear')
        x_p3 = torch.exp(-log_x_p3_0 + r3_1).clamp(min=0, max=1)

        r4_0 = F.upsample(self.p4_0(f), size=x0.size()[2:], mode='bilinear')
        # r4_1 = F.upsample(self.p4_1(f), size=x0.size()[2:], mode='bilinear')
        x_enhance = local_contrast_enhancement(x0)
        x_p4 = ((x_enhance+r4_0+x)* self.std_out + self.mean_out).clamp(min=0., max=1.)
        #x_p4 = (0.1 * x_enhance+x+r4_0).clamp(min=0., max=1.)
        
        attention_fusion = F.upsample(self.attentional_fusion(f), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 5, :, :], 1) * torch.stack(
            (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :], x_p4[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
                                                                                                     x_p1[:, 1, :, :],
                                                                                                     x_p2[:, 1, :, :],
                                                                                                     x_p3[:, 1, :, :],
                                                                                                     x_p4[:, 1, :, :]),
                                                                                                    1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
                                                                                                   x_p1[:, 2, :, :],
                                                                                                   x_p2[:, 2, :, :],
                                                                                                   x_p3[:, 2, :, :],
                                                                                                    x_p4[:, 2, :, :]),
                                                                                                  1), 1, True)),
                             1).clamp(min=0, max=1)

        # if self.training:
        #     return x_fusion, x_p0, x_p1, x_p2, x_p3, x_p4
        # else:
        #     return x_fusion
        return x_fusion, x_p0, x_p1, x_p2, x_p3, x_p4, r0, r1, r2_0, r2_1, r3_0, r3_1, r4_0