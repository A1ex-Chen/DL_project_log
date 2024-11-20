def forward(self, x):
    if self.mode == 'constant':
        return self.homographies
    else:
        return self.homographies[self.image_count]
