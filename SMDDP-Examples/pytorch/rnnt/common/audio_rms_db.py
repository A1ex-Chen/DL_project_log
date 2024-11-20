@property
def rms_db(self):
    mean_square = np.mean(self._samples ** 2)
    return 10 * np.log10(mean_square)
