def write_results(self, i, p, im, s):
    """Write inference results to a file or directory."""
    string = ''
    if len(im.shape) == 3:
        im = im[None]
    if (self.source_type.stream or self.source_type.from_img or self.
        source_type.tensor):
        string += f'{i}: '
        frame = self.dataset.count
    else:
        match = re.search('frame (\\d+)/', s[i])
        frame = int(match[1]) if match else None
    self.txt_path = self.save_dir / 'labels' / (p.stem + ('' if self.
        dataset.mode == 'image' else f'_{frame}'))
    string += '%gx%g ' % im.shape[2:]
    result = self.results[i]
    result.save_dir = self.save_dir.__str__()
    string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
    if self.args.save or self.args.show:
        self.plotted_img = result.plot(line_width=self.args.line_width,
            boxes=self.args.show_boxes, conf=self.args.show_conf, labels=
            self.args.show_labels, im_gpu=None if self.args.retina_masks else
            im[i])
    if self.args.save_txt:
        result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
    if self.args.save_crop:
        result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.
            txt_path.stem)
    if self.args.show:
        self.show(str(p))
    if self.args.save:
        self.save_predicted_images(str(self.save_dir / p.name), frame)
    return string
