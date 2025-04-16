def spectral_normalize(self, magnitudes, normalize_fun):
    output = dynamic_range_compression(magnitudes, normalize_fun)
    return output
