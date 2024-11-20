def embed_detections(self, detections: Sequence[Detection], image: (Image.
    Image | np.ndarray)) ->np.ndarray:
    if isinstance(detections, Detections):
        detections = detections.to_list()
    if isinstance(image, Image.Image):
        image = np.array(image)
    box_images = []
    for detection in detections:
        if isinstance(detection, Detection):
            x, y, w, h = list(map(int, detection.box))
        elif isinstance(detection, dict):
            x, y, w, h = list(map(int, detection['box']))
        else:
            raise ValueError(f'Unknown detection type: {type(detection)}')
        box_image = image[y:y + h, x:x + w]
        box_images.append(box_image)
    with torch.no_grad():
        z = self(box_images)
    return z
