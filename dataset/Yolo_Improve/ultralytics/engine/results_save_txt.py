def save_txt(self, txt_file, save_conf=False):
    """
        Save detection results to a text file.

        Args:
            txt_file (str): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Example:
            ```python
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model("path/to/image.jpg")
            for result in results:
                result.save_txt("output.txt")
            ```

        Notes:
            - The file will contain one line per detection or classification with the following structure:
                - For detections: `class confidence x_center y_center width height`
                - For classifications: `confidence class_name`
                - For masks and keypoints, the specific formats will vary accordingly.

            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.

            - Existing contents of the file will not be overwritten; new results will be appended.
        """
    is_obb = self.obb is not None
    boxes = self.obb if is_obb else self.boxes
    masks = self.masks
    probs = self.probs
    kpts = self.keypoints
    texts = []
    if probs is not None:
        [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in
            probs.top5]
    elif boxes:
        for j, d in enumerate(boxes):
            c, conf, id = int(d.cls), float(d.conf
                ), None if d.id is None else int(d.id.item())
            line = c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1))
            if masks:
                seg = masks[j].xyn[0].copy().reshape(-1)
                line = c, *seg
            if kpts is not None:
                kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2
                    ) if kpts[j].has_visible else kpts[j].xyn
                line += *kpt.reshape(-1).tolist(),
            line += (conf,) * save_conf + (() if id is None else (id,))
            texts.append(('%g ' * len(line)).rstrip() % line)
    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, 'a') as f:
            f.writelines(text + '\n' for text in texts)
