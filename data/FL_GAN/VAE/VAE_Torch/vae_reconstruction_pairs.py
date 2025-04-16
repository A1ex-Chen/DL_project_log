def reconstruction_pairs(no_reconstructions=50):
    """
        Creating reconstruction pairs (x, x') where x is the original image and x' is the decoder-output
        """
    x_original = test_data[:no_reconstructions]
    x_dequantized = dequantize(x_original, dequantize=False)
    x_reconstructed = vae(x_dequantized)[0]
    x_reconstructed = dequantize(x_reconstructed, reverse=True)
    pairs = torch.zeros_like(torch.cat((x_original, x_reconstructed), dim=0)
        ).detach().cpu().numpy()
    pairs[::2] = x_original.detach().cpu().numpy()
    pairs[1::2] = x_reconstructed.detach().cpu().numpy()
    pairs = np.clip(pairs, 0, 255)
    return np.transpose(pairs, [0, 2, 3, 1])
