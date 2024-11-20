def show(self, **kwargs) ->Image.Image:
    im = self.im
    boxes = self.preds[:, :4]
    labels = [f'{int(c)} {conf:.2f}' for conf, c in self.preds[:, 4:]]
    draw_im = draw_bounding_boxes(im, boxes, labels, **kwargs)
    return Image.fromarray(draw_im)
