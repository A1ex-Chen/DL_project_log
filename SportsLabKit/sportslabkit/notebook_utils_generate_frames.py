def generate_frames(ground_truth_positions, width=360, height=240, box_size=25
    ):
    frames = []
    for gt_positions in ground_truth_positions:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, (x, y, w, h) in enumerate(gt_positions):
            box_color = np.array([idx ** 2 * 30 % 256, (idx ** 2 * 50 + 50) %
                256, (idx ** 2 * 20 + 100) % 256], dtype=np.uint8)
            noise = np.random.randint(-20, 20, (box_size, box_size, 3),
                dtype=np.int16)
            noisy_box_color = np.clip(box_color + noise, 0, 255).astype(np.
                uint8)
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, width), min(y + h, height)
            if y2 - y1 > 0 and x2 - x1 > 0:
                frame[y1:y2, x1:x2] = noisy_box_color[:y2 - y1, :x2 - x1]
        frames.append(frame)
    return frames
