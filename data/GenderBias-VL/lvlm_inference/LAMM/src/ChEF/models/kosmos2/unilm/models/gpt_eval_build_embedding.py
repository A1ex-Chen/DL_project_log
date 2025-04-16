@classmethod
def build_embedding(cls, args, dictionary, embed_dim, path=None):
    return Embedding(len(dictionary), embed_dim, dictionary.pad())
