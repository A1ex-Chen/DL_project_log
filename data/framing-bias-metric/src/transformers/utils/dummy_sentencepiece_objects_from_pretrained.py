@classmethod
def from_pretrained(self, *args, **kwargs):
    requires_sentencepiece(self)
