def forward(self, c5):
    p6 = self.p6(c5)
    p7 = self.p7(F.relu(p6))
    return [p6, p7]
