def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
        output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
        output = self.main(input)
    return output.view(-1, 1).squeeze(1)
