def forward(self, x):
    return self.w_2(self.dropout(self.activation(self.w_1(x))))
