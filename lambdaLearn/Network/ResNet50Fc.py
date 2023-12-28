import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import ModuleList
from torchvision import models


def to_var(x, requires_grad=True, device="cuda:0"):
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


class ResNet50Fc(nn.Module):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, num_classes, bias=True, output_feature=False):
        super(ResNet50Fc, self).__init__()
        _model_resnet = models.resnet50(pretrained=True)
        model_resnet = _model_resnet
        self.conv1 = model_resnet.conv1
        self.bn_source = model_resnet.bn1
        self.num_classes = num_classes
        # self.bn_target = copy.deepcopy(self.bn_source)
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        __in_features = model_resnet.fc.in_features
        self.output_feature = output_feature
        self.bottleneck = nn.Linear(__in_features, 256)
        self.bias = bias
        self.relu_bottle = nn.ReLU()
        if isinstance(self.num_classes, (list, tuple)):
            self.fc = ModuleList([])
            for i in range(len(self.num_classes)):
                self.fc.append(nn.Linear(256, self.num_classes[i], bias=self.bias[i]))
        else:
            self.fc = nn.Linear(256, self.num_classes, bias=self.bias)
        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x, source=False):
        x = self.conv1(x)
        # if source:
        x = self.bn_source(x)
        # else:
        # x = self.bn_target(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.relu_bottle(x)
        if isinstance(self.fc, ModuleList):
            output = []
            for c in self.fc:
                output.append(c(x))
        else:
            output = self.fc(x)
        if self.output_feature:
            return x, output
        else:
            return output

    # def rigister(self, curr_mod, name, param,device):
    #     if '.' in name:
    #         n = name.split('.')
    #         module_name = n[0]
    #         rest = '.'.join(n[1:])
    #         for name, mod in curr_mod.named_children():
    #             if module_name == name:
    #                 self.rigister(mod, rest, param,device)
    #                 break
    #     else:
    #         if name=='weight' or name=='bias':
    #             print('except')
    #             # grads = torch.autograd.grad(Parameter(param), (grad), create_graph=True, allow_unused=True)
    #             # print(grads)
    #             try:
    #                 curr_mod.register_buffer(name,to_var(getattr(curr_mod,name).data,device=device))
    #             except:
    #                 print(curr_mod)
    #             setattr(curr_mod, name, param)

    def update_params(
        self,
        lr_inner,
        first_order=False,
        source_params=None,
        detach=False,
        device="cuda:0",
    ):
        # for tgt in self.named_parameters():
        #     name_t, param_t = tgt
        #     self.rigister(self, name_t, param_t,device)
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data, device=device)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp, grad, device=device)
        else:
            for name, param in self.named_parameters():
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data, device=device)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp, device=device)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param, device=device)

    def set_param(self, curr_mod, name, param, grad=None, device="cuda:0"):
        if "." in name:
            n = name.split(".")
            module_name = n[0]
            rest = ".".join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            # try:
            setattr(curr_mod, name, param)
            # except:
            #     print(curr_mod)
            #     # getattr(curr_mod, name).data=param
            #     curr_mod._buffers[name] = to_var(getattr(curr_mod, name).data, device=device)
            #     curr_mod._non_persistent_buffers_set.discard(name)
            #     # print('except')
            #     # grads = torch.autograd.grad(Parameter(param), (grad), create_graph=True, allow_unused=True)
            #     # print(grads)
            #     # curr_mod.register_buffer(name,getattr(curr_mod,name).data)
            #     setattr(curr_mod, name, param)

    def output_num(self):
        return self.__in_features


#
# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
# class BasicBlock(nn.Module):
#     expansion: int = 1
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion: int = 4
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(
#         self,
#         block,
#         layers,
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation= None,
#         norm_layer= None
#     ) -> None:
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#
#     def _make_layer(self, block, planes: int, blocks: int,
#                     stride: int = 1, dilate: bool = False) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#     def forward(self, x) :
#         return self._forward_impl(x)
