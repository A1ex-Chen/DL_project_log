def get_file_format(self, seed, shape):
    return (
        f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"
        )
