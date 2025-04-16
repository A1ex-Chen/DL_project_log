def forward(self, x):
    x = self.stem(x)
    for i in range(self.num_layers):
        fn = getattr(self, f'layer{i + 1}')
        x = fn(x)
    x = self.classifier(x)
    return x
