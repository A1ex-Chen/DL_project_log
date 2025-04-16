def forward(self, inputs):
    C3, C4, C5 = inputs
    P5 = self.P5(C5)
    up5 = self.P5_upsampled(P5)
    concat1 = self.concat([up5, C4])
    conv1 = self.conv1(concat1)
    P4 = self.P4(conv1)
    up4 = self.P4_upsampled(P4)
    concat2 = self.concat([C3, up4])
    PP3 = self.P3(concat2)
    return PP3, P4, P5
