def forward(self, x):
    outputs = []
    x = self.conv_0(x)
    x = self.lite_effiblock_1(x)
    x = self.lite_effiblock_2(x)
    outputs.append(x)
    x = self.lite_effiblock_3(x)
    outputs.append(x)
    x = self.lite_effiblock_4(x)
    outputs.append(x)
    return tuple(outputs)
