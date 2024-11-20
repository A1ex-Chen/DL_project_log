def make_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1,
        times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)
    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))
        x_start = np.random.randint(int(margin), resolution - int(margin) -
            width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) -
            height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0
    mask = 1 - mask if random.random() < 0.5 else mask
    return mask
