def __init__(self, kernel):
    self.tid = kernel['tid']
    self.device = kernel['device']
    self.stream = kernel['stream']
    self.grid = str(kernel['grid']).replace(' ', '').replace('(', '').replace(
        ')', '')
    self.block = str(kernel['block']).replace(' ', '').replace('(', ''
        ).replace(')', '')
    self.name = kernel['kShortName'].replace(' ', '_')
    self.lName = kernel['kLongName']
    self.sil = kernel['kDuration']
    self.index = None
    self.argMarker = kernel['marker']
    self.modMarker = kernel['reprMarkers']
    self.seqMarker = kernel['seqMarker']
    self.layer = kernel['layer']
    self.trace = kernel['trace']
    self.seqId = kernel['seqId']
    self.altSeqId = kernel['altSeqId']
    self.dir = kernel['dir']
    self.sub = kernel['subSeqId']
    self.mod = 'na'
    self.op = 'na'
    self.params = {'na': 'na'}
    self.tc = 'na'
    self.flops = 0
    self.bytes = 0
