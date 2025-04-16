def get_dummy_seed_input(self, batch_size=1, embedding_dim=768,
    num_embeddings=77, seed=0):
    torch.manual_seed(seed)
    batch_size = batch_size
    embedding_dim = embedding_dim
    num_embeddings = num_embeddings
    hidden_states = torch.randn((batch_size, embedding_dim)).to(torch_device)
    proj_embedding = torch.randn((batch_size, embedding_dim)).to(torch_device)
    encoder_hidden_states = torch.randn((batch_size, num_embeddings,
        embedding_dim)).to(torch_device)
    return {'hidden_states': hidden_states, 'timestep': 2, 'proj_embedding':
        proj_embedding, 'encoder_hidden_states': encoder_hidden_states}
