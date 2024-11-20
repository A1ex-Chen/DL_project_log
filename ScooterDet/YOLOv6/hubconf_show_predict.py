def show_predict(self, img_path, min_score=0.5, figsize=(16, 16), color=
    'lawngreen', linewidth=2):
    prediction = self.predict(img_path)
    boxes, scores, classes = prediction['boxes'], prediction['scores'
        ], prediction['classes']
    visualize_detections(Image.open(img_path), boxes, classes, scores,
        min_score=min_score, figsize=figsize, color=color, linewidth=linewidth)
