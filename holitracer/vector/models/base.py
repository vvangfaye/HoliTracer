import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import transformer
import numpy as np

from .backbone.swin_transformer import swin_b, swin_l, swin_b_224
from .module.snake import Snake
from .module.vlr_offset import VLROffset
from .module.vlr_cls import VLRCls
from .utils import img_poly_to_can_poly, compute_polygon_angles, compute_polygon_angles_with_loop, compute_polygon_angles_with_loop_n_neighbors


class PointRefinementModel(nn.Module):
    def __init__(self, num_points=32):
        super(PointRefinementModel, self).__init__()
        self.num_points = num_points
        # 图像特征提取网络
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入通道数为3（RGB图像）
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
        )
        # 全连接层，将图像特征展平
        self.fc_image = nn.Linear(256, 256)
        # 处理预测点的全连接层
        self.fc_points = nn.Linear(num_points * 2, 256)
        # 综合图像和预测点特征
        self.fc_combined = nn.Linear(512, 256)
        # 输出位移差值（回归）和角点分类（分类）
        self.fc_displacements = nn.Linear(256, num_points * 2)  # 位移差值输出
        self.fc_is_corner = nn.Linear(256, num_points)  # 角点分类输出

    def forward(self, image, pred_points):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        """
        batch_size = image.size(0)
        # 图像特征
        x_image = self.conv_layers(image)
        x_image = x_image.view(batch_size, -1)
        x_image = F.relu(self.fc_image(x_image))
        # 预测点特征
        x_points = pred_points.view(batch_size, -1)  # 展平
        x_points = F.relu(self.fc_points(x_points))
        # 综合特征
        x = torch.cat((x_image, x_points), dim=1)
        x = F.relu(self.fc_combined(x))
        # 输出
        displacements = self.fc_displacements(x)
        displacements = displacements.view(batch_size, self.num_points, 2)
        is_corner_logits = self.fc_is_corner(x)
        return displacements, is_corner_logits


def get_points_feature(img_feature, img_poly, h, w):
    # img_feature: CNN 特征图 (B, C, H, W)
    # img_poly: 多边形坐标 (B, num_points, 2)
    # 克隆 img_poly 以避免修改原始数据
    img_poly = img_poly.clone()

    # 归一化多边形坐标到 [-1, 1] 范围
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.0) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.0) - 1

    # 获取维度信息
    batch_size, num_channels, _, _ = img_feature.size()
    num_polygons, num_points, _ = img_poly.size()

    # 初始化 gcn_feature 张量
    gcn_feature = torch.zeros(
        batch_size, num_channels, num_points, device=img_poly.device
    )

    # 遍历每个批次
    for i in range(batch_size):
        # 获取当前批次的多边形和 CNN 特征
        current_poly = img_poly[i].unsqueeze(0).unsqueeze(2)
        current_feature = img_feature[i].unsqueeze(0)
        # current_poly: (1, num_points, 1, 2)
        # current_feature: (1, C, H, W)

        # 使用 grid_sample 进行特征采样
        sampled_feature = F.grid_sample(
            current_feature, current_poly, align_corners=True
        )
        # sampled_feature: (1, C, num_points, 1)

        # 调整维度并赋值给 gcn_feature
        gcn_feature[i] = sampled_feature.squeeze(0).squeeze(2)
    # gcn_feature: (B, C, num_points)
    return gcn_feature


class PointRefinementModel2(nn.Module):
    def __init__(self, num_points, backbone_path, down_ratio=4):
        super(PointRefinementModel2, self).__init__()
        self.num_points = num_points
        self.down_ratio = down_ratio
        self.stride = 4
        self.backbone_path = backbone_path
        self.backbone = swin_l(pretrained=True)

        # backbone 冻结
        for param in self.backbone.parameters():
            param.requires_grad = False

        tranformer_feature_size = 192
        self.trans_feature = nn.Sequential(
            nn.Conv2d(
                tranformer_feature_size, 256, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.trans_poly = nn.Linear(
            in_features=num_points * 64, out_features=num_points * 4, bias=False
        )
        self.trans_fuse = nn.Linear(
            in_features=num_points * 4, out_features=num_points * 2, bias=True
        )

        self.trans_feature_cls = nn.Sequential(
            nn.Conv2d(
                tranformer_feature_size, 256, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.cls_poly = nn.Linear(
            in_features=num_points * 64, out_features=num_points, bias=True
        )
        # 输出1个值来表示二分类概率
        self.cls_fuse = nn.Linear(
            in_features=num_points, out_features=num_points, bias=True
        )

        # 不再需要Sigmoid激活函数，因为BCEWithLogitsLoss会自动处理
        self.load_pretrained_weights(self.backbone_path)

    def load_pretrained_weights(self, backbone_path):
        if backbone_path:
            state = torch.load(backbone_path)
            self.load_state_dict(state, strict=False)

    def forward(self, image, pred_points):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        """
        # 图像特征, backbone 冻结
        with torch.no_grad():
            transformer_features = self.backbone(image)[0]

        # transformer_features = self.backbone(image)[0]

        poly_num = pred_points.size(0)
        h, w = transformer_features.size(2), transformer_features.size(3)

        feature = self.trans_feature(transformer_features)
        points_feature = get_points_feature(feature, pred_points, h, w).view(
            poly_num, -1
        )

        points_feature = self.trans_poly(points_feature)
        offsets = self.trans_fuse(points_feature).view(poly_num, self.num_points, 2)
        refined_points = offsets * self.stride + pred_points.detach()
        final_refined_points = refined_points * self.down_ratio

        cls_feature = self.trans_feature_cls(transformer_features)
        cls_points_feature = get_points_feature(cls_feature, refined_points, h, w).view(
            poly_num, -1
        )
        cls_points = self.cls_poly(cls_points_feature)
        cls_points = self.cls_fuse(cls_points).view(poly_num, self.num_points)
        # cls_points 的输出是每个点的二分类 logits (B, num_points)

        return final_refined_points, cls_points


class SnakeRefineModel(nn.Module):
    def __init__(self, num_points, backbone_path, snake_num=3, down_ratio=4):
        super(SnakeRefineModel, self).__init__()
        self.num_points = num_points
        self.down_ratio = down_ratio
        self.stride = 4
        self.backbone_path = backbone_path
        self.backbone = swin_l(pretrained=True)

        # backbone 冻结
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.snake_num = snake_num
        for i in range(snake_num):
            evolve_gcn = Snake(256, 64 + 2, conv_type="dgrid")
            self.__setattr__("evolve_gcn" + str(i), evolve_gcn)

        tranformer_feature_size = 192
        self.trans_feature = nn.Sequential(
            nn.Conv2d(
                tranformer_feature_size, 256, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.trans_feature_cls = nn.Sequential(
            nn.Conv2d(
                tranformer_feature_size, 256, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.cls_poly = nn.Linear(
            in_features=num_points * 64, out_features=num_points, bias=True
        )
        # 输出1个值来表示二分类概率
        self.cls_fuse = nn.Linear(
            in_features=num_points, out_features=num_points, bias=True
        )

        # 不再需要Sigmoid激活函数，因为BCEWithLogitsLoss会自动处理
        self.load_pretrained_weights(self.backbone_path)

    def load_pretrained_weights(self, backbone_path):
        if backbone_path:
            state = torch.load(backbone_path)
            self.load_state_dict(state, strict=False)

    def _snake_evolve(
        self,
        snake,
        transformer_feature,
        i_it_poly,
        c_it_poly,
        valid_mask,
        stride=1.0,
        ignore=False,
    ):
        h, w = transformer_feature.size(2), transformer_feature.size(3)
        init_feature = get_points_feature(
            transformer_feature, i_it_poly, h, w
        ) * valid_mask.unsqueeze(2)
        # init_feature shape: B, 64, num_points
        # c_it_poly shape: B, num_points, 2
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = snake(init_input).permute(0, 2, 1) * valid_mask.unsqueeze(2)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward(self, image, pred_points, valid_mask=None):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        """
        # 图像特征, backbone 冻结
        with torch.no_grad():
            transformer_features = self.backbone(image)[0]

        # transformer_features = self.backbone(image)[0]

        poly_num = pred_points.size(0)
        h, w = transformer_features.size(2), transformer_features.size(3)

        poly_features = self.trans_feature(transformer_features)

        py_pred = pred_points
        for i in range(self.snake_num):
            c_py_pred = img_poly_to_can_poly(py_pred, valid_mask)
            evolve_gcn = self.__getattr__("evolve_gcn" + str(i))
            py_pred = self._snake_evolve(
                evolve_gcn,
                poly_features,
                py_pred,
                c_py_pred,
                valid_mask,
                stride=self.stride,
            )

        cls_features = self.trans_feature_cls(transformer_features)
        cls_points_features = get_points_feature(
            cls_features, py_pred, h, w
        ) * valid_mask.unsqueeze(2)

        cls_points_features = cls_points_features.view(poly_num, -1)
        cls_points = self.cls_poly(cls_points_features)
        cls_points = (
            self.cls_fuse(cls_points).view(poly_num, self.num_points) * valid_mask
        )
        # cls_points 的输出是每个点的二分类 logits (B, num_points)
        final_refined_points = py_pred * self.down_ratio

        return final_refined_points, cls_points


class VLRModel(nn.Module):
    def __init__(self, num_points, backbone_path, vlr_num=3, down_ratio=4):
        super(VLRModel, self).__init__()
        self.num_points = num_points
        self.down_ratio = down_ratio
        self.stride = 4
        self.backbone_path = backbone_path
        self.backbone = swin_l(pretrained=True)

        # backbone 冻结
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.vlr_num = vlr_num
        for i in range(vlr_num):
            evolve_vlr = VLROffset(256, 192 + 2, num_points=num_points)
            self.__setattr__("evolve_vlr" + str(i), evolve_vlr)

        self.vlr_cls = VLRCls(256, 192 + 2, num_points=num_points)

        self.load_pretrained_weights(self.backbone_path)

    def load_pretrained_weights(self, backbone_path):
        if backbone_path:
            state = torch.load(backbone_path)
            self.load_state_dict(state, strict=False)

    def _model_evolve(
        self,
        model,
        transformer_feature,
        i_it_poly,
        c_it_poly,
        valid_mask,
        stride=1.0,
        ignore=False,
    ):
        h, w = transformer_feature.size(2), transformer_feature.size(3)
        init_feature = get_points_feature(
            transformer_feature, i_it_poly, h, w
        ) * valid_mask.unsqueeze(1)
        # init_feature shape: B, 64, num_points
        # c_it_poly shape: B, num_points, 2
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = model(init_input).permute(0, 2, 1) * valid_mask.unsqueeze(2)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward(self, image, pred_points, valid_mask=None):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        """
        # 图像特征, backbone 冻结
        with torch.no_grad():
            transformer_features = self.backbone(image)[0]

        poly_num = pred_points.size(0)
        h, w = transformer_features.size(2), transformer_features.size(3)

        py_pred = pred_points
        for i in range(self.vlr_num):
            c_py_pred = img_poly_to_can_poly(py_pred, valid_mask)
            evolve_vlr = self.__getattr__("evolve_vlr" + str(i))
            py_pred = self._model_evolve(
                evolve_vlr,
                transformer_features,
                py_pred,
                c_py_pred,
                valid_mask,
                stride=self.stride,
            )

        cls_points_features = get_points_feature(
            transformer_features, py_pred, h, w
        ) * valid_mask.unsqueeze(1)

        cls_features = torch.cat([cls_points_features, py_pred.permute(0, 2, 1)], dim=1)
        cls_points = self.vlr_cls(cls_features, valid_mask) * valid_mask

        final_refined_points = py_pred * self.down_ratio

        return final_refined_points, cls_points

class VLRAModel(nn.Module):
    def __init__(self, num_points, backbone_path, vlr_num=3, down_ratio=4):
        super(VLRAModel, self).__init__()
        self.num_points = num_points
        self.down_ratio = down_ratio
        self.stride = 4
        self.backbone_path = backbone_path
        self.backbone = swin_l(pretrained=True)

        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.vlr_num = vlr_num
        for i in range(vlr_num):
            evolve_vlr = VLROffset(256, 192 + 2, num_points=num_points)
            self.__setattr__("evolve_vlr" + str(i), evolve_vlr)

        # +2 +1 拼接 angle 特征
        self.vlr_cls = VLRCls(256, 192 + 3, num_points=num_points)

        self.load_pretrained_weights(self.backbone_path)

    def load_pretrained_weights(self, backbone_path):
        if backbone_path:
            state = torch.load(backbone_path)
            self.load_state_dict(state, strict=False)

    def _model_evolve(
        self,
        model,
        transformer_feature,
        i_it_poly,
        c_it_poly,
        valid_mask,
        stride=1.0,
        ignore=False,
    ):
        h, w = transformer_feature.size(2), transformer_feature.size(3)
        init_feature = get_points_feature(
            transformer_feature, i_it_poly, h, w
        ) * valid_mask.unsqueeze(1)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = model(init_input).permute(0, 2, 1) * valid_mask.unsqueeze(2)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward(self, image, pred_points, valid_mask=None):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        corner_labels: (B, num_points), 1表示角点，0表示非角点
        compute_loss: 是否计算并返回angle loss
        """
        # 1) 提取 backbone 特征
        with torch.no_grad():
            transformer_features = self.backbone(image)[0]
        h, w = transformer_features.size(2), transformer_features.size(3)

        # 2) 多次offset回归
        py_pred = pred_points
        for i in range(self.vlr_num):
            c_py_pred = img_poly_to_can_poly(py_pred, valid_mask)
            evolve_vlr = self.__getattr__("evolve_vlr" + str(i))
            py_pred = self._model_evolve(
                evolve_vlr,
                transformer_features,
                py_pred,
                c_py_pred,
                valid_mask,
                stride=self.stride,
            )

        # 3) 分类输入特征
        cls_points_features = get_points_feature(
            transformer_features, py_pred, h, w
        ) * valid_mask.unsqueeze(1)

        # 4) 计算角度 & 拼接
        with torch.no_grad():
            angles = compute_polygon_angles_with_loop(py_pred, valid_mask)
            angles_1d = angles.unsqueeze(1)          # (B, 1, N)

        cls_features = torch.cat([
            cls_points_features,          # (B, 192, N)
            py_pred.permute(0, 2, 1),     # (B, 2,   N)
            angles_1d                     # (B, 1,   N)
        ], dim=1)

        # 5) 分类输出
        cls_points = self.vlr_cls(cls_features, valid_mask)  # => shape (B, N)

        # 6) 回归后的多边形点（乘down_ratio）
        final_refined_points = py_pred * self.down_ratio

        return final_refined_points, cls_points


class VLRAsModel(nn.Module):
    def __init__(self, num_points, backbone_path, vlr_num=3, down_ratio=4):
        super(VLRAsModel, self).__init__()
        self.num_points = num_points
        self.down_ratio = down_ratio
        self.stride = 4
        self.backbone_path = backbone_path
        self.backbone = swin_l(pretrained=True)

        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.vlr_num = vlr_num
        for i in range(vlr_num):
            evolve_vlr = VLROffset(256, 192 + 2, num_points=num_points)
            self.__setattr__("evolve_vlr" + str(i), evolve_vlr)

        # +2 +3 拼接 angle 特征
        self.vlr_cls = VLRCls(256, 192 + 5, num_points=num_points)

        self.load_pretrained_weights(self.backbone_path)

    def load_pretrained_weights(self, backbone_path):
        if backbone_path:
            state = torch.load(backbone_path)
            self.load_state_dict(state, strict=False)

    def _model_evolve(
        self,
        model,
        transformer_feature,
        i_it_poly,
        c_it_poly,
        valid_mask,
        stride=1.0,
        ignore=False,
    ):
        h, w = transformer_feature.size(2), transformer_feature.size(3)
        init_feature = get_points_feature(
            transformer_feature, i_it_poly, h, w
        ) * valid_mask.unsqueeze(1)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = model(init_input).permute(0, 2, 1) * valid_mask.unsqueeze(2)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward(self, image, pred_points, valid_mask=None):
        """
        image: (B, 3, H, W)
        pred_points: (B, num_points, 2)
        corner_labels: (B, num_points), 1表示角点，0表示非角点
        compute_loss: 是否计算并返回angle loss
        """
        # 1) 提取 backbone 特征
        with torch.no_grad():
            transformer_features = self.backbone(image)[0]
        h, w = transformer_features.size(2), transformer_features.size(3)

        # 2) 多次offset回归
        py_pred = pred_points
        for i in range(self.vlr_num):
            c_py_pred = img_poly_to_can_poly(py_pred, valid_mask)
            evolve_vlr = self.__getattr__("evolve_vlr" + str(i))
            py_pred = self._model_evolve(
                evolve_vlr,
                transformer_features,
                py_pred,
                c_py_pred,
                valid_mask,
                stride=self.stride,
            )

        # 3) 分类输入特征
        cls_points_features = get_points_feature(
            transformer_features, py_pred, h, w
        ) * valid_mask.unsqueeze(1)

        # 4) 计算角度 & 拼接
        with torch.no_grad():
            angles1 = compute_polygon_angles_with_loop_n_neighbors(py_pred, valid_mask, 1).unsqueeze(1)
            angles2 = compute_polygon_angles_with_loop_n_neighbors(py_pred, valid_mask, 2).unsqueeze(1)
            angles3 = compute_polygon_angles_with_loop_n_neighbors(py_pred, valid_mask, 3).unsqueeze(1)

        # draw py_pred
        # numpy_py_pred = py_pred[0].detach().cpu().numpy()
        # img = np.zeros((h, w, 3), dtype=np.uint8)
        # for i in range(py_pred.size(1)):
        #     x, y = int(numpy_py_pred[i][0]), int(numpy_py_pred[i][1])
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        # cv2.imwrite("py_pred.png", img)

        cls_features = torch.cat([
            cls_points_features,          # (B, 192, N)
            py_pred.permute(0, 2, 1),     # (B, 2,   N)
            angles1,                      # (B, 1,   N)
            angles2,                      # (B, 1,   N)
            angles3                       # (B, 1,   N)
        ], dim=1)

        # 5) 分类输出
        cls_points = self.vlr_cls(cls_features, valid_mask)  # => shape (B, N)

        # 6) 回归后的多边形点（乘down_ratio）
        final_refined_points = py_pred * self.down_ratio

        return final_refined_points, cls_points
