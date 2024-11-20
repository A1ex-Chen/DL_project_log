def setDetParams(self):
    self.imgIds = []
    self.catIds = []
    self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)
        ) + 1, endpoint=True)
    self.recThrs = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) +
        1, endpoint=True)
    self.maxDets = [1, 10, 100]
    self.areaRng = [[0 ** 2, 100000.0 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 
        96 ** 2], [96 ** 2, 100000.0 ** 2]]
    self.areaRngLbl = ['all', 'small', 'medium', 'large']
    self.useCats = 1
