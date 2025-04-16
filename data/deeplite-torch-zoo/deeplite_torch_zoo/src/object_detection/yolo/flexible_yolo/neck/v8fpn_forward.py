def forward(self, inputs):
    C3, C4, C5 = inputs
    up5 = self.P5_upsampled(C5)
    concat1 = self.concat([up5, C4])
    P4 = self.conv1(concat1)
    up4 = self.P4_upsampled(P4)
    concat2 = self.concat([C3, up4])
    PP3 = self.P3(concat2)
    return PP3, P4, C5
