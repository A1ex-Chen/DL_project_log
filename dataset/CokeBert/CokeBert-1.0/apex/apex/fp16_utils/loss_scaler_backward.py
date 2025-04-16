def backward(self, loss, retain_graph=False):
    scaled_loss = loss * self.loss_scale
    scaled_loss.backward(retain_graph=retain_graph)
