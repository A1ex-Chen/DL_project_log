def forward(self, input):
    """
        VGG-11 for CIFAR-10
        @innpv size (32, 3, 32, 32)
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
    output = self.conv4(output)
    output = self.bn4(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = self.conv5(output)
    output = self.bn5(output)
    output = self.relu(output)
    output = self.conv6(output)
    output = self.bn6(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = self.conv7(output)
    output = self.bn7(output)
    output = self.relu(output)
    output = self.conv8(output)
    output = self.bn8(output)
    output = self.relu(output)
    output = self.max_pool(output)
    output = output.view(-1, 512)
    output = self.linear(output)
    return output