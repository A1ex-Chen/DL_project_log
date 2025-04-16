def __init__(self, model, classesFile, p6=False, confThreshold=0.5,
    nmsThreshold=0.5, objThreshold=0.5):
    with open(classesFile, 'rt') as f:
        self.class_names = f.read().rstrip('\n').split('\n')
    self.net = cv2.dnn.readNet(model)
    self.input_size = 640, 640
    self.mean = 0.485, 0.456, 0.406
    self.std = 0.229, 0.224, 0.225
    if not p6:
        self.strides = [8, 16, 32]
    else:
        self.strides = [8, 16, 32, 64]
    self.confThreshold = confThreshold
    self.nmsThreshold = nmsThreshold
    self.objThreshold = objThreshold
