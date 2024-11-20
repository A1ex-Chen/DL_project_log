def forward(self, input):
    input = input.view(input.size(0), -1)
    return self.model.forward(input)
