def simulate_moving_object(num_frames: int, x: int=0, y: int=125, vx: int=
    15, vy: int=0, box_size: int=25, class_id: int=1):
    box_position = np.array([x, y])
    box_velocity = np.array([vx, vy])
    detections = []
    ground_truth_positions = []
    for _ in range(num_frames):
        score = random.uniform(0.9, 1.0)
        noise_scaling_factor = 10 / score
        box_position += box_velocity
        box = [box_position[0] - box_size // 2, box_position[1] - box_size //
            2, box_size, box_size]
        noisy_box_position = box_position + np.random.normal(0,
            noise_scaling_factor, 2)
        noisy_box = [int(noisy_box_position[0] - box_size // 2), int(
            noisy_box_position[1] - box_size // 2), box_size, box_size]
        det: Detection = Detection(box=noisy_box, score=score, class_id=
            class_id)
        detections.append(det)
        ground_truth_positions.append(box)
    return detections, ground_truth_positions
