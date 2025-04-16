def save_crop(self, save_dir, file_name=Path('im.jpg')):
    """
        Save cropped detection images to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Directory path where the cropped images should be saved.
            file_name (str | pathlib.Path): Filename for the saved cropped image.

        Notes:
            This function does not support Classify or Oriented Bounding Box (OBB) tasks. It will warn and exit if
            called for such tasks.

        Example:
            ```python
            from ultralytics import YOLO

            model = YOLO("yolov8n.pt")
            results = model("path/to/image.jpg")

            # Save cropped images to the specified directory
            for result in results:
                result.save_crop(save_dir="path/to/save/crops", file_name="crop")
            ```
        """
    if self.probs is not None:
        LOGGER.warning('WARNING ⚠️ Classify task do not support `save_crop`.')
        return
    if self.obb is not None:
        LOGGER.warning('WARNING ⚠️ OBB task do not support `save_crop`.')
        return
    for d in self.boxes:
        save_one_box(d.xyxy, self.orig_img.copy(), file=Path(save_dir) /
            self.names[int(d.cls)] / f'{Path(file_name)}.jpg', BGR=True)
