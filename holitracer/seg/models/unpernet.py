# ========================================
# File Name: unpernet.py
# From article: https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/
# Created Date: 2024-11-22 22:21:08
# Description: The UnperNet model for semantic segmentation.
# ========================================


from .base import BaseNet
import torch
from torch import nn
import torch.nn.functional as F
from .module.batchnorm import SynchronizedBatchNorm2d  # , PrRoIPool2D
from .module.GCM import Global_Context_Module


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        SynchronizedBatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class UPerNet(BaseNet):

    def __init__(self, backbone, nclass,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fc_dim=4096,
                 fpn_dim: int = 512, pretrain: bool = False, isContext: bool = False):
        super(UPerNet, self).__init__(backbone, pretrain)
        fc_dim = self.backbone.channels[-1]
        fpn_inplanes = self.backbone.channels
        # print(self.backbone.channels)
        self.nclass = nclass

        # Global_Context_Module
        self.isContext = isContext
        # print(f"isContext = {self.isContext}")
        if self.isContext:
            self.GCM = Global_Context_Module(channels=self.backbone.channels)

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            # self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))  # TODO
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))

            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        # input: Fusion out, input_dim: fpn_dim
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nclass, kernel_size=1, bias=True)
        )

    def base_forward0(self, x):
        seg_size = x.shape[-2:]
        conv_out = self.backbone.base_forward(x)
        # for idx, f in enumerate(conv_out):
        #     print(idx, f.shape)
        conv5 = conv_out[-1]
        input_size = conv5.size()
        # TODO
        # roi = []  # fake rois, just used for pooling
        # for i in range(input_size[0]):  # batch size
        #     roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))  # b, x0, y0, x1, y1
        # roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            # print(pool_conv)
            # print(pool_scale(conv5).shape)
            ppm_out.append(pool_conv(F.interpolate(
                # pool_scale(conv5, roi.detach()), #TODO
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        # for idx, f in enumerate(ppm_out):
        #     print(idx, f.shape)
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)
        out = self.object_head(x)
        # print("out::", out.shape)
        return F.interpolate(out, size=seg_size, mode='bilinear', align_corners=False)

    def base_forward_context(self, xs):
        x1, x2, x3, save_name = xs
        with torch.no_grad():
            f2 = self.backbone.base_forward(x2)
            f3 = self.backbone.base_forward(x3)
        f1 = self.backbone.base_forward(x1)

        seg_size = x1.shape[-2:]
        conv_out = self.GCM(f1, f2, f3, save_name)

        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                # pool_scale(conv5, roi.detach()), #TODO
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))

        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)
        out = self.object_head(x)
        return F.interpolate(out, size=seg_size, mode='bilinear', align_corners=False)

    def base_forward(self, x):
        if self.isContext:
            return self.base_forward_context(x)
        else:
            return self.base_forward0(x)
