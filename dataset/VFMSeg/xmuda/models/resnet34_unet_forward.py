def forward(self, x):
    h, w = x.shape[2], x.shape[3]
    min_size = 16
    pad_h = int((h + min_size - 1) / min_size) * min_size - h
    pad_w = int((w + min_size - 1) / min_size) * min_size - w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [0, pad_w, 0, pad_h])
    inter_features = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    inter_features.append(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    inter_features.append(x)
    x = self.layer2(x)
    inter_features.append(x)
    x = self.layer3(x)
    x = self.dropout(x)
    inter_features.append(x)
    x = self.layer4(x)
    x = self.dropout(x)
    x = self.dec_t_conv_stage5(x)
    x = torch.cat([inter_features[3], x], dim=1)
    x = self.dec_conv_stage4(x)
    x = self.dec_t_conv_stage4(x)
    x = torch.cat([inter_features[2], x], dim=1)
    x = self.dec_conv_stage3(x)
    x = self.dec_t_conv_stage3(x)
    x = torch.cat([inter_features[1], x], dim=1)
    x = self.dec_conv_stage2(x)
    x = self.dec_t_conv_stage2(x)
    x = torch.cat([inter_features[0], x], dim=1)
    x = self.dec_conv_stage1(x)
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, 0:h, 0:w]
    return x
