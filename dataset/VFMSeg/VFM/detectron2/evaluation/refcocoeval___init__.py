def __init__(self, iouType='segm'):
    if iouType == 'segm' or iouType == 'bbox':
        self.setDetParams()
    elif iouType == 'keypoints':
        self.setKpParams()
    else:
        raise Exception('iouType not supported')
    self.iouType = iouType
    self.useSegm = None
