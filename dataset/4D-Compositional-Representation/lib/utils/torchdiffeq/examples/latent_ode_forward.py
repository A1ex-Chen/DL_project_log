def forward(self, z):
    out = self.fc1(z)
    out = self.relu(out)
    out = self.fc2(out)
    return out
