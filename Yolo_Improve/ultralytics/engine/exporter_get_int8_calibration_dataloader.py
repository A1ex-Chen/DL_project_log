def get_int8_calibration_dataloader(self, prefix=''):
    """Build and return a dataloader suitable for calibration of INT8 models."""
    LOGGER.info(
        f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'"
        )
    data = (check_cls_dataset if self.model.task == 'classify' else
        check_det_dataset)(self.args.data)
    dataset = YOLODataset(data[self.args.split or 'val'], data=data, task=
        self.model.task, imgsz=self.imgsz[0], augment=False, batch_size=
        self.args.batch * 2)
    n = len(dataset)
    if n < 300:
        LOGGER.warning(
            f'{prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.'
            )
    return build_dataloader(dataset, batch=self.args.batch * 2, workers=0)
