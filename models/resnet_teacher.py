import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['resnet34_dermnet', 'resnet34_dermnet_aux', 'resnet50_dermnet', 'resnet50_dermnet_aux']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, teacher_feature=None):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        f1_t_resize = x
        B1, C1, W1, H1 = f1.shape
        f1 = f1.view(B1, C1 * W1 * H1)
        asj1 = f1.mm(f1.t())
        if teacher_feature:
            x = torch.matmul(f1_t_resize.permute(1, 2, 3, 0), teacher_feature[0]).permute(3, 0, 1, 2)
            # x = teacher_feature[0].mm(f1).reshape(B1, C1, W1, H1)
        out1 = x

        x = self.layer2(x)
        f2 = x
        f2_t_resize = x
        B1, C2, W2, H2 = f2.shape
        f2 = f2.view(B1, C2 * W2 * H2)
        asj2 = f2.mm(f2.t())
        if teacher_feature:
            x = torch.matmul(f2_t_resize.permute(1, 2, 3, 0), teacher_feature[1]).permute(3, 0, 1, 2)
            # x = teacher_feature[1].mm(f2).reshape(B1, C2, W2, H2)
        out2 = x

        x = self.layer3(x)
        f3 = x
        f3_t_resize = x
        B1, C3, W3, H3 = f3.shape
        f3 = f3.view(B1, C3 * W3 * H3)
        asj3 = f3.mm(f3.t())
        if teacher_feature:
            x = torch.matmul(f3_t_resize.permute(1, 2, 3, 0), teacher_feature[2]).permute(3, 0, 1, 2)
            # x = teacher_feature[2].mm(f3).reshape(B1, C3, W3, H3)
        out3 = x

        x = self.layer4(x)
        f4 = x
        f4_t_resize = x
        B1, C4, W4, H4 = f4.shape
        f4 = f4.view(B1, C4 * W4 * H4)
        asj4 = f4.mm(f4.t())
        d = 8
        if teacher_feature:
            x = torch.matmul(f4_t_resize.permute(1, 2, 3, 0), teacher_feature[3]).permute(3, 0, 1, 2)
            # x = teacher_feature[3].mm(f4).reshape(B1, C4, W4, H4)
        out4 = x

        branch1_out = F.normalize(out1.pow(2).mean(1).view(out1.size(0), -1))
        branch2_out = F.normalize(out2.pow(2).mean(1).view(out2.size(0), -1))
        branch3_out = F.normalize(out3.pow(2).mean(1).view(out3.size(0), -1))
        branch4_out = F.normalize(out4.pow(2).mean(1).view(out4.size(0), -1))

        x = self.avgpool(x)
        out = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if is_feat:
            return [out1, out2, out3, out4], [asj1, asj2, asj3, asj4, x, out], [branch1_out, branch2_out, branch3_out, branch4_out]
        else:
            return x

class Auxiliary_Classifier(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Auxiliary_Classifier, self).__init__()

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, 128, layers[1], stride=2),
                                                self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 128 * block.expansion
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 256 * block.expansion
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 512 * block.expansion
        self.block_extractor4 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=1)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            mid = out
            mid = mid.mm(mid.t())
            ss_logits.append(mid)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class ResNet_Auxiliary(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.auxiliary_classifier = Auxiliary_Classifier(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.logsigma1 = nn.Parameter(torch.FloatTensor(1 / 64 * torch.ones((64, 1))))

    def forward(self, x, grad=False, teacher_feature=None):
        if grad is False:
            feats, logit, att = self.backbone(x, is_feat=True, teacher_feature=None)
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        else:
            feats, logit, att = self.backbone(x, is_feat=True, teacher_feature=teacher_feature)

        ss_logits = self.auxiliary_classifier(feats)

        return feats, att, logit, [ss_logits[0], ss_logits[2], ss_logits[4], ss_logits[6]], [ss_logits[1], ss_logits[3],ss_logits[5],ss_logits[7]], self.logsigma1


def resnet34_dermnet(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet34_dermnet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_dermnet(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet50_dermnet_aux(**kwargs):
    return ResNet_Auxiliary(Bottleneck, [3, 4, 6, 3], **kwargs)
