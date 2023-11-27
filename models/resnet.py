import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet18_dermnet', 'resnet18_dermnet_aux']


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

        self.layerwise_num_channels = [64, 128, 256, 512]
        self.teacher_layerwise_num_channels = [64, 128, 256, 512]
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[0], self.teacher_layerwise_num_channels[0], kernel_size=3,stride=1, padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[0], self.teacher_layerwise_num_channels[0], kernel_size=3,stride=1, padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
            conv1x1(self.teacher_layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[1], self.teacher_layerwise_num_channels[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[1], self.teacher_layerwise_num_channels[1], kernel_size=3, stride=1, padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
            conv1x1(self.teacher_layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[2], self.teacher_layerwise_num_channels[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[2], self.teacher_layerwise_num_channels[2], kernel_size=3, stride=1, padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
            conv1x1(self.teacher_layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[3], self.teacher_layerwise_num_channels[3], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[3], self.teacher_layerwise_num_channels[3], kernel_size=3, stride=1, padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[3], self.teacher_layerwise_num_channels[3]),
            conv1x1(self.teacher_layerwise_num_channels[3], self.teacher_layerwise_num_channels[3]),
        )

        # variance are represented as a softplus function applied to "variance parameters".
        init_variance_param_value = self._variance_param_to_variance(torch.tensor(5.0))
        self.variance_param1 = nn.Parameter(
            torch.full([3136], init_variance_param_value)
        )
        self.variance_param2 = nn.Parameter(
            torch.full([784], init_variance_param_value)
        )
        self.variance_param3 = nn.Parameter(
            torch.full([196], init_variance_param_value)
        )
        self.variance_param4 = nn.Parameter(
            torch.full([49], init_variance_param_value)
        )

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

    def _varance_to_variance_param(self, variance):
        """
        Convert variance to corresponding variance parameter by inverse of the softplus function.
        :param torch.FloatTensor variance: the target variance for obtaining the variance parameter
        """
        return torch.log(torch.exp(variance) - 1.0)

    def _variance_param_to_variance(self, variance_param):
        """
        Convert the variance parameter to corresponding variance by the softplus function.
        :param torch.FloatTensor variance_param: the target variance parameter for obtaining the variance
        """
        return torch.log(torch.exp(variance_param) + 1.0)

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

        input = x

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
        if teacher_feature:
            x = torch.matmul(f4_t_resize.permute(1, 2, 3, 0), teacher_feature[3]).permute(3, 0, 1, 2)
            # x = teacher_feature[3].mm(f4).reshape(B1, C4, W4, H4)
        out4 = x

        x = self.avgpool(x)
        out = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        a1 = torch.cuda.FloatTensor(64, 64, 56, 56).uniform_() > 0.6
        o1 = out1.masked_fill(a1, 0)
        a2 = torch.cuda.FloatTensor(64, 128, 28, 28).uniform_() > 0.6
        o2 = out2.masked_fill(a2, 0)
        a3 = torch.cuda.FloatTensor(64, 256, 14, 14).uniform_() > 0.6
        o3 = out3.masked_fill(a3, 0)
        a4 = torch.cuda.FloatTensor(64, 512, 7, 7).uniform_() > 0.6
        o4 = out4.masked_fill(a4, 0)
        branch1_out = self.branch1(o1)
        branch2_out = self.branch2(o2)
        branch3_out = self.branch3(o3)
        branch4_out = self.branch4(o4)

        branch1_out = F.normalize(branch1_out.pow(2).mean(1).view(branch1_out.size(0), -1))
        branch2_out = F.normalize(branch2_out.pow(2).mean(1).view(branch2_out.size(0), -1))
        branch3_out = F.normalize(branch3_out.pow(2).mean(1).view(branch3_out.size(0), -1))
        branch4_out = F.normalize(branch4_out.pow(2).mean(1).view(branch4_out.size(0), -1))


        variance1 = self._variance_param_to_variance(self.variance_param1)
        variance2 = self._variance_param_to_variance(self.variance_param2)
        variance3 = self._variance_param_to_variance(self.variance_param3)
        variance4 = self._variance_param_to_variance(self.variance_param4)

        # If the input has an additional dimension for mini-batch, resize the variance to match its dimension
        if input.dim() == 4:
            variance1 = variance1.unsqueeze(0)
            variance2 = variance2.unsqueeze(0)
            variance3 = variance3.unsqueeze(0)
            variance4 = variance4.unsqueeze(0)

        if is_feat:
            return x, [(branch1_out, variance1), (branch2_out, variance2), (branch3_out, variance3), (branch4_out, variance4)], [out1, out2, out3, out4]
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

        # variance are represented as a softplus function applied to "variance parameters".
        init_variance_param_value = self._variance_param_to_variance(torch.tensor(5.0))
        self.variance_param = nn.Parameter(
            torch.full([64], init_variance_param_value)
        )
    def _varance_to_variance_param(self, variance):
        """
        Convert variance to corresponding variance parameter by inverse of the softplus function.
        :param torch.FloatTensor variance: the target variance for obtaining the variance parameter
        """
        return torch.log(torch.exp(variance) - 1.0)

    def _variance_param_to_variance(self, variance_param):
        """
        Convert the variance parameter to corresponding variance by the softplus function.
        :param torch.FloatTensor variance_param: the target variance parameter for obtaining the variance
        """
        return torch.log(torch.exp(variance_param) + 1.0)

    def forward(self, x, grad=False, teacher_feature=None):
        if grad is False:
            logit, mid, feats = self.backbone(x, is_feat=True, teacher_feature=None)
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        else:
            logit, mid, feats = self.backbone(x, is_feat=True, teacher_feature=None)

        variance1 = self._variance_param_to_variance(self.variance_param)
        variance2 = self._variance_param_to_variance(self.variance_param)
        variance3 = self._variance_param_to_variance(self.variance_param)
        variance4 = self._variance_param_to_variance(self.variance_param)

        # If the input has an additional dimension for mini-batch, resize the variance to match its dimension
        if x.dim() == 4:
            variance1 = variance1.unsqueeze(0)
            variance2 = variance2.unsqueeze(0)
            variance3 = variance3.unsqueeze(0)
            variance4 = variance4.unsqueeze(0)

        ss_logits = self.auxiliary_classifier(feats)

        return logit, mid, feats, [(ss_logits[0], variance1), (ss_logits[2], variance2), (ss_logits[4], variance3), (ss_logits[6], variance4)], [ss_logits[1], ss_logits[3], ss_logits[5], ss_logits[7]], self.logsigma1, ss_logits


def resnet18_dermnet(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_dermnet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [2, 2, 2, 2], **kwargs)
