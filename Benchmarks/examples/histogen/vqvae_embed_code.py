def embed_code(self, embed_id):
    return F.embedding(embed_id, self.embed.transpose(0, 1))
