def forward(self, x):
    return self.mapper(self.backbone(x))
