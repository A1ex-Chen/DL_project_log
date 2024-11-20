@property
def dummy_input(self):
    batch_size = 4
    embedding_dim = 8
    num_embeddings = 7
    hidden_states = floats_tensor((batch_size, embedding_dim)).to(torch_device)
    proj_embedding = floats_tensor((batch_size, embedding_dim)).to(torch_device
        )
    encoder_hidden_states = floats_tensor((batch_size, num_embeddings,
        embedding_dim)).to(torch_device)
    return {'hidden_states': hidden_states, 'timestep': 2, 'proj_embedding':
        proj_embedding, 'encoder_hidden_states': encoder_hidden_states}
