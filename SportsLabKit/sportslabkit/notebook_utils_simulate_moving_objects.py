def simulate_moving_objects(num_objects: int, num_frames: int, width: int=
    360, height: int=240, box_size: int=25, frame_drop_rate: float=0.1):
    all_detections: list[list[Detection]] = [[] for _ in range(num_frames)]
    all_gt_positions: list[list[list[int]]] = [[] for _ in range(num_frames)]
    for obj in range(num_objects):
        x = random.randint(0, width - box_size)
        y = random.randint(0, height - box_size)
        vx = random.randint(-5, 5)
        vy = random.randint(-5, 5)
        if x < width / 2:
            vx = random.randint(0, 5)
        else:
            vx = random.randint(-5, 0)
        if y < height / 2:
            vy = random.randint(0, 5)
        else:
            vy = random.randint(-5, 0)
        detections, ground_truth_positions = simulate_moving_object(num_frames,
            x, y, vx, vy, box_size, class_id=obj)
        for frame in range(num_frames):
            det = detections[frame]
            gt_pos = ground_truth_positions[frame]
            all_gt_positions[frame].append(gt_pos)
            if random.random() < frame_drop_rate:
                continue
            all_detections[frame].append(det)
    return all_detections, all_gt_positions
