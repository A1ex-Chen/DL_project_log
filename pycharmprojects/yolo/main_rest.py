import torch
import torchvision
from efficientnet_pytorch import EfficientNet
# vit = torchvision.models.vit_l_16()
# # efficientnet_b7 = torchvision.models.efficientnet_b7()
# print(vit)
# x = torch.randn(1, 3, 240, 360)
# for i, feature in enumerate(vit.image_size):
#     x = feature(x)
#     print(i, x.shape)
convnext_base = torchvision.models.convnext_base(pretrained=False)
print(convnext_base)
conv1 = convnext_base.features[:2]
conv2 = convnext_base.features[2:4]
conv3 = convnext_base.features[4:6]
conv4 = convnext_base.features[6:]
# print(conv4)
x = torch.randn(1, 3, 224, 224)
for i in range(len(convnext_base.features)):
    x = convnext_base.features[i](x)
    print(i, x.shape)
# x = torch.randn(1, 3, 224, 224)
# for i in range(len(regnet_y_8gf.stem)):
#     x = regnet_y_8gf.stem[i](x)
#     print(i, x.shape)
# for i in range(len(regnet_y_8gf.trunk_output)):
#     x = regnet_y_8gf.trunk_output[i](x)
#     print(i, x.shape)