def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):

    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)
