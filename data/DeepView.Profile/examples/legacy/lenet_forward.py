def forward(self, input):
    """
        LeNet for CIFAR-10
        @innpv size (512, 3, 32, 32)
        """
    output = self.conv1(input)
    output = self.tanh(output)
    output = self.pool(output)
    output = self.conv2(output)
    output = self.tanh(output)
    output = self.pool(output)
    output = output.view(-1, 1250)
    output = self.dense1(output)
    output = self.tanh(output)
    output = self.dense2(output)
    output = self.softmax(output)
    return output
