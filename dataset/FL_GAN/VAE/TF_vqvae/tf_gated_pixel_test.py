def test(self, images, with_update=False):
    if with_update:
        _, cost = self.sess.run([self.optim, self.loss], {self.inputs:
            images[0], self.target_pixels: images[1]})
    else:
        cost = self.sess.run(self.loss, {self.inputs: images[0], self.
            target_pixels: images[1]})
    return cost
