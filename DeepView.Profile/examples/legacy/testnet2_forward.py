def forward(self, input):
    """
        @innpv size (16, 3, 128, 128)
        """
    output = self.conv1(input)
    output = self.bn1(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = self.conv2(output)
    output = self.bn2(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = self.conv3(output)
    output = self.bn3(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = self.conv4(output)
    output = self.bn4(output)
    output = self.relu(output)
    output = self.max_pool2(output)
    output = output.view(output.size(0), -1)
    output = self.linear(output)
    return output