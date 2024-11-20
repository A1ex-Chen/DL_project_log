def preprocess_batch(self, batch):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch['img'] = batch['img'].to(self.device, non_blocking=True).float(
        ) / 255
    if self.args.multi_scale:
        imgs = batch['img']
        sz = random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 +
            self.stride) // self.stride * self.stride
        sf = sz / max(imgs.shape[2:])
        if sf != 1:
            ns = [(math.ceil(x * sf / self.stride) * self.stride) for x in
                imgs.shape[2:]]
            imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear',
                align_corners=False)
        batch['img'] = imgs
    return batch
