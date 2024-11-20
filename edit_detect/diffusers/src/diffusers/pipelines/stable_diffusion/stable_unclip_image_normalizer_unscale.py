def unscale(self, embeds):
    embeds = embeds * self.std + self.mean
    return embeds
