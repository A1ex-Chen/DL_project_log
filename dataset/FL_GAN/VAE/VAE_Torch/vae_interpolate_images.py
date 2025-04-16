def interpolate_images(no_interpolations=10):
    interpolations = torch.zeros(size=(no_interpolations * 10, C, H, W)).float(
        ).cuda()
    counter = 0
    weights = np.linspace(0, 1, 10)[1:-1]
    for i in range(no_interpolations):
        x_a, x_b = test_data[i].unsqueeze(0), test_data[i + no_interpolations
            ].unsqueeze(0)
        x_a_dequantized, x_b_dequantized = dequantize(x_a, dequantize=False
            ), dequantize(x_b, dequantize=False)
        z_a, z_b = vae(x_a_dequantized)[2], vae(x_b_dequantized)[2]
        interpolations[counter] = x_a
        counter += 1
        for weight in weights:
            z_interpolated = (1 - weight) * z_a + weight * z_b
            x_interpolated = vae.decoder(z_interpolated)
            x_interpolated = dequantize(x_interpolated, reverse=True)
            interpolations[counter] = x_interpolated[0]
            counter += 1
        interpolations[counter] = x_b
        counter += 1
    interpolations = interpolations.detach().cpu().numpy()
    interpolations = np.clip(interpolations, 0, 255)
    return np.transpose(interpolations, [0, 2, 3, 1])
