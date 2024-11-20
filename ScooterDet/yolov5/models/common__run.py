def _run(self, pprint=False, show=False, save=False, crop=False, render=
    False, labels=True, save_dir=Path('')):
    s, crops = '', []
    for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
        s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            s = s.rstrip(', ')
            if show or save or render or crop:
                annotator = Annotator(im, example=str(self.names))
                for *box, conf, cls in reversed(pred):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    if crop:
                        file = save_dir / 'crops' / self.names[int(cls)
                            ] / self.files[i] if save else None
                        crops.append({'box': box, 'conf': conf, 'cls': cls,
                            'label': label, 'im': save_one_box(box, im,
                            file=file, save=save)})
                    else:
                        annotator.box_label(box, label if labels else '',
                            color=colors(cls))
                im = annotator.im
        else:
            s += '(no detections)'
        im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray
            ) else im
        if show:
            if is_jupyter():
                from IPython.display import display
                display(im)
            else:
                im.show(self.files[i])
        if save:
            f = self.files[i]
            im.save(save_dir / f)
            if i == self.n - 1:
                LOGGER.info(
                    f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}"
                    )
        if render:
            self.ims[i] = np.asarray(im)
    if pprint:
        s = s.lstrip('\n')
        return (
            f"""{s}
Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}"""
             % self.t)
    if crop:
        if save:
            LOGGER.info(f'Saved results to {save_dir}\n')
        return crops
