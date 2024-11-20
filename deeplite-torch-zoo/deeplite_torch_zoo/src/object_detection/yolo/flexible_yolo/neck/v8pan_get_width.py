def get_width(self, n):
    return make_divisible(n * self.gw, self.channel_divisor)
