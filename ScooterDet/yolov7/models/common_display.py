def display(self, pprint=False, show=False, save=False, render=False,
    save_dir=''):
    colors = color_list()
    for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
        str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()
                str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            if show or save or render:
                for *box, conf, cls in pred:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(box, img, label=label, color=colors[int(
                        cls) % 10])
        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.
            ndarray) else img
        if pprint:
            print(str.rstrip(', '))
        if show:
            img.show(self.files[i])
        if save:
            f = self.files[i]
            img.save(Path(save_dir) / f)
            print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else
                f' to {save_dir}\n')
        if render:
            self.imgs[i] = np.asarray(img)
