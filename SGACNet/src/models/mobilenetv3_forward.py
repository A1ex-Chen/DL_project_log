def forward(self, x):
    x = self.features(x)
    x = x.flatten(1)
    if self.drop_rate > 0.0:
        x = F.dropout(x, p=self.drop_rate, training=self.training)
    return self.classifier(x)
