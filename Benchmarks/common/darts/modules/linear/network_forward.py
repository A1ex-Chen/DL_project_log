def forward(self, x):
    s0 = s1 = self.stem(x)
    for i, cell in enumerate(self.cells):
        if cell.reduction:
            weights = F.softmax(self.alpha_reduce, dim=-1)
        else:
            weights = F.softmax(self.alpha_normal, dim=-1)
        s0, s1 = s1, cell(s0, s1, weights)
    logits = self.classifier(s1.view(s1.size(0), -1))
    return logits
