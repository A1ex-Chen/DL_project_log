def setKernelInfo(self, info):
    self.kNameId = info['name']
    self.corrId = int(info['correlationId'])
    start = int(info['start'])
    end = int(info['end'])
    assert end > start, 'This assertion can fail for very large profiles. It usually fails when start = end = 0.'
    self.kStartTime = start
    self.kEndTime = end
    self.kDuration = end - start
    assert start > Kernel.profStart
    self.device = int(info['deviceId'])
    self.stream = int(info['streamId'])
    self.grid = info['gridX'], info['gridY'], info['gridZ']
    self.block = info['blockX'], info['blockY'], info['blockZ']
    self.timeOffset = Kernel.profStart
