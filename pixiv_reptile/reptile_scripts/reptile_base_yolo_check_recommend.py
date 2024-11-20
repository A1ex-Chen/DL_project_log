def yolo_check_recommend(self, source):
    model = YOLO('../other/pose-best.pt')
    results = model.predict(source=source, imgsz=640, half=True)
    recommend = False
    for result in tqdm(results):
        if recommend:
            return recommend
        keypoints = result.keypoints
        if keypoints.conf.shape != torch.Size([0]):
            points = keypoints.conf[0]
            recommend = recommend or self.check_is_recommend(points)
    return recommend
