def __call__(self, im, labels, p=1.0):
    if self.transform and random.random() < p:
        new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=
            labels[:, 0])
        im, labels = new['image'], np.array([[c, *b] for c, b in zip(new[
            'class_labels'], new['bboxes'])])
    return im, labels
