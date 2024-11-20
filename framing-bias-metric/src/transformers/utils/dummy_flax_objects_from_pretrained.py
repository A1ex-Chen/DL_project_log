@classmethod
def from_pretrained(self, *args, **kwargs):
    requires_flax(self)
