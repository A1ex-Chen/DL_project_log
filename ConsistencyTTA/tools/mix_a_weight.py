def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq) - np
        .log10(freq_sq + 12194 ** 2) - np.log10(freq_sq + 20.6 ** 2) - 0.5 *
        np.log10(freq_sq + 107.7 ** 2) - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)
    return weight
