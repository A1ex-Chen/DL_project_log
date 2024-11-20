def __init__(self, token_num=40, emb_dim=384):
    super().__init__()
    self.embedding = torch.nn.Embedding(token_num, emb_dim)
    self.trans = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
        torch.nn.GELU(), torch.nn.Linear(emb_dim, emb_dim))
