def general_augment(self, img, labels):
    """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
    nl = len(labels)
    augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'],
        vgain=self.hyp['hsv_v'])
    if random.random() < self.hyp['flipud']:
        img = np.flipud(img)
        if nl:
            labels[:, 2] = 1 - labels[:, 2]
    if random.random() < self.hyp['fliplr']:
        img = np.fliplr(img)
        if nl:
            labels[:, 1] = 1 - labels[:, 1]
    return img, labels
