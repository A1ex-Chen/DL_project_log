def forward(self, input_data):
    self.magnitude, self.phase = self.transform(input_data)
    reconstruction = self.inverse(self.magnitude, self.phase)
    return reconstruction
