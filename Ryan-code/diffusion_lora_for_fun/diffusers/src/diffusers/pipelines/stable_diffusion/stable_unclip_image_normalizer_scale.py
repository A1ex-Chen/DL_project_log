def scale(self, embeds):
    embeds = (embeds - self.mean) * 1.0 / self.std
    return embeds
