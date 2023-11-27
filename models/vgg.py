'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

__all__ = ['vgg13_bn_aux', 'vgg13_bn']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layerwise_num_channels = [128, 256, 512, 512]
        self.teacher_layerwise_num_channels = [256, 512, 1024, 2048]
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[0], self.layerwise_num_channels[0], kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[0], self.layerwise_num_channels[0], kernel_size=3, stride=1,
                      padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
            conv1x1(self.teacher_layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[1], self.layerwise_num_channels[1], kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[1], self.layerwise_num_channels[1], kernel_size=3, stride=1,
                      padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
            conv1x1(self.teacher_layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[2], self.layerwise_num_channels[2], kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[2], self.layerwise_num_channels[2], kernel_size=3, stride=1,
                      padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
            conv1x1(self.teacher_layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(self.layerwise_num_channels[3], self.layerwise_num_channels[3], kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.Conv2d(self.layerwise_num_channels[3], self.layerwise_num_channels[3], kernel_size=3, stride=1,
                      padding=1, bias=True),
            conv1x1(self.layerwise_num_channels[3], self.teacher_layerwise_num_channels[3]),
            conv1x1(self.teacher_layerwise_num_channels[3], self.teacher_layerwise_num_channels[3]),
        )
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

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

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

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        x = F.relu(self.block0(x))
        f0 = x 

        x = self.pool0(x)
        f0_1 = x
        x = self.block1(x)
        x = F.relu(x)
        f1 = x  

        x = self.pool1(x)
        f1_1 = x
        x = self.block2(x)
        x = F.relu(x)
        f2 = x  

        x = self.pool2(x)
        f2_1 = x
        x = self.block3(x)
        x = F.relu(x)
        f3 = x  

        x = self.pool3(x)
        f3_1 = x
        x = self.block4(x)
        x = F.relu(x)
        f4 = x  

        x = self.pool4(x)
        out = x  

        x = self.avgpool(out)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        a1 = torch.cuda.FloatTensor(64, 128, 56, 56).uniform_() > 0.6
        o1 = f1_1.masked_fill(a1, 0)
        a2 = torch.cuda.FloatTensor(64, 256, 28, 28).uniform_() > 0.6
        o2 = f2_1.masked_fill(a2, 0)
        a3 = torch.cuda.FloatTensor(64, 512, 14, 14).uniform_() > 0.6
        o3 = f3_1.masked_fill(a3, 0)
        a4 = torch.cuda.FloatTensor(64, 512, 7, 7).uniform_() > 0.6
        o4 = out.masked_fill(a4, 0)
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

        if input.dim() == 4:
            variance1 = variance1.unsqueeze(0)
            variance2 = variance2.unsqueeze(0)
            variance3 = variance3.unsqueeze(0)
            variance4 = variance4.unsqueeze(0)


        return x, [(branch1_out, variance1), (branch2_out, variance2), (branch3_out, variance3),
                   (branch4_out, variance4)], [f1_1, f2_1, f3_1, out]

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Auxiliary_Classifier(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=100):
        super(Auxiliary_Classifier, self).__init__()

        self.block_extractor1 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[2], batch_norm, cfg[1][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor2 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor3 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor4 = nn.Sequential(*[self._make_layers(cfg[4], batch_norm, cfg[4][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = out.view(-1, 512)
            mid = out
            mid = mid.mm(mid.t())
            ss_logits.append(mid)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class VGG_Auxiliary(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG_Auxiliary, self).__init__()
        self.backbone = VGG(cfg, batch_norm=batch_norm, num_classes=num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(cfg, batch_norm=batch_norm, num_classes=num_classes)
        self.logsigma = nn.Parameter(torch.FloatTensor(1 / 64 * torch.ones((64, 1))))

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


    def forward(self, x, grad=False):
        logit, mid, feats = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)

        variance1 = self._variance_param_to_variance(self.variance_param)
        variance2 = self._variance_param_to_variance(self.variance_param)
        variance3 = self._variance_param_to_variance(self.variance_param)
        variance4 = self._variance_param_to_variance(self.variance_param)

        if x.dim() == 4:
            variance1 = variance1.unsqueeze(0)
            variance2 = variance2.unsqueeze(0)
            variance3 = variance3.unsqueeze(0)
            variance4 = variance4.unsqueeze(0)

        return logit, mid, feats, [(ss_logits[0], variance1), (ss_logits[2], variance2), (ss_logits[4], variance3),
                                   (ss_logits[6], variance4)], [ss_logits[1], ss_logits[3], ss_logits[5], ss_logits[7]], self.logsigma1, ss_logits

cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}

def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg13_bn_aux(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_Auxiliary(cfg['B'], batch_norm=True, **kwargs)
    return model
