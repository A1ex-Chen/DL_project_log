def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output
