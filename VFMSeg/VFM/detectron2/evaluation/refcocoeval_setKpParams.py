def setKpParams(self):
    self.imgIds = []
    self.catIds = []
    self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)
        ) + 1, endpoint=True)
    self.recThrs = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) +
        1, endpoint=True)
    self.maxDets = [20]
    self.areaRng = [[0 ** 2, 100000.0 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 
        100000.0 ** 2]]
    self.areaRngLbl = ['all', 'medium', 'large']
    self.useCats = 1
    self.kpt_oks_sigmas = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 
        0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
        ) / 10.0
