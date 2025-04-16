def _make_classifier(self, num_features, num_classes, dropout):
    return nn.Sequential(OrderedDict([('pooling', nn.AdaptiveAvgPool2d(1)),
        ('squeeze', LambdaLayer(lambda x: x.squeeze(-1).squeeze(-1))), (
        'dropout', nn.Dropout(dropout)), ('fc', nn.Linear(num_features,
        num_classes))]))
