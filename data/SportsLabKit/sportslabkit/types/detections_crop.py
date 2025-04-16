def crop(self, save: bool=True, save_dir: (str | Path)='runs/detect/exp',
    exist_ok: bool=False) ->list[Image.Image]:
    save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
    images = []
    for box in self.preds[:, :4]:
        cropped_im = self.im[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        if save_dir is not None:
            Image.fromarray(cropped_im).save(Path(save_dir) / f'{box}.png')
        images.append(cropped_im)
    return images
