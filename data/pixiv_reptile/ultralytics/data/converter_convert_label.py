def convert_label(image_name, image_width, image_height, orig_label_dir,
    save_dir):
    """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
    orig_label_path = orig_label_dir / f'{image_name}.txt'
    save_path = save_dir / f'{image_name}.txt'
    with orig_label_path.open('r') as f, save_path.open('w') as g:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_name = parts[8]
            class_idx = class_mapping[class_name]
            coords = [float(p) for p in parts[:8]]
            normalized_coords = [(coords[i] / image_width if i % 2 == 0 else
                coords[i] / image_height) for i in range(8)]
            formatted_coords = ['{:.6g}'.format(coord) for coord in
                normalized_coords]
            g.write(f"{class_idx} {' '.join(formatted_coords)}\n")
