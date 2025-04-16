@cached_property
def config(self):
    config = MarianConfig.from_pretrained('sshleifer/tiny-marian-en-de')
    return config
