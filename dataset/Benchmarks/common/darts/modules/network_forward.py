def forward(self, x):
    s0 = s1 = self.stem(x)
    for i, cell in enumerate(self.cells):
        weights = F.softmax(self.alpha_normal, dim=-1)
        s0, out = s1, cell(s0, s1, weights)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits
