def forward(self, x):
    features = {}
    remaining_features = self.features_list.copy()
    x = x.unsqueeze(1)
    x = self.Conv2d_1a_3x3(x)
    x = self.Conv2d_2a_3x3(x)
    x = self.Conv2d_2b_3x3(x)
    x = self.maxpool1(x)
    if '64' in remaining_features:
        features['64'] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        remaining_features.remove('64')
        if len(remaining_features) == 0:
            return tuple(features[a] for a in self.features_list)
    x = self.Conv2d_3b_1x1(x)
    x = self.Conv2d_4a_3x3(x)
    x = self.maxpool2(x)
    if '192' in remaining_features:
        features['192'] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        remaining_features.remove('192')
        if len(remaining_features) == 0:
            return tuple(features[a] for a in self.features_list)
    x = self.Mixed_5b(x)
    x = self.Mixed_5c(x)
    x = self.Mixed_5d(x)
    x = self.Mixed_6a(x)
    x = self.Mixed_6b(x)
    x = self.Mixed_6c(x)
    x = self.Mixed_6d(x)
    x = self.Mixed_6e(x)
    if '768' in remaining_features:
        features['768'] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        remaining_features.remove('768')
        if len(remaining_features) == 0:
            return tuple(features[a] for a in self.features_list)
    x = self.Mixed_7a(x)
    x = self.Mixed_7b(x)
    x = self.Mixed_7c(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    x = torch.flatten(x, 1)
    if '2048' in remaining_features:
        features['2048'] = x
        remaining_features.remove('2048')
        if len(remaining_features) == 0:
            return tuple(features[a] for a in self.features_list)
    if 'logits_unbiased' in remaining_features:
        x = x.mm(self.fc.weight.T)
        features['logits_unbiased'] = x
        remaining_features.remove('logits_unbiased')
        if len(remaining_features) == 0:
            return tuple(features[a] for a in self.features_list)
        x = x + self.fc.bias.unsqueeze(0)
    else:
        x = self.fc(x)
    features['logits'] = x
    return tuple(features[a] for a in self.features_list)
