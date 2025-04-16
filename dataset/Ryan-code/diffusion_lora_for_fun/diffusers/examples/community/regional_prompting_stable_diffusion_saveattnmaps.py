def saveattnmaps(self, output, h, w, th, step, regions):
    masks = []
    for i, mask in enumerate(self.history[step].values()):
        img, _, mask = makepmask(self, mask, h, w, th[i % len(th)], step)
        if self.ex:
            masks = [(x - mask) for x in masks]
            masks.append(mask)
            if len(masks) == regions - 1:
                output.images.extend([FF.to_pil_image(mask) for mask in masks])
                masks = []
        else:
            output.images.append(img)
