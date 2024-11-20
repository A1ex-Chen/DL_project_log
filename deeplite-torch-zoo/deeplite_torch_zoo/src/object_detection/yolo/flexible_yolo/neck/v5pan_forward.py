def forward(self, inputs):
    PP3, P4, P5 = inputs
    convp3 = self.convP3(PP3)
    concat3_4 = self.concat([convp3, P4])
    PP4 = self.P4(concat3_4)
    convp4 = self.convP4(PP4)
    concat4_5 = self.concat([convp4, P5])
    PP5 = self.P5(concat4_5)
    return PP3, PP4, PP5
