def forward(self, x):
    x = self.bert(x)
    return self.out(x)
