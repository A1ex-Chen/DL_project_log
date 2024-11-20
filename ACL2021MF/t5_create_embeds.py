def create_embeds(model, target_vocab):
    vocab_emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    vec_dim = vocab_emb.shape[1]
    w_list = [None for _ in range(len(target_vocab))]
    for index in target_vocab:
        t = target_vocab[index]
        vec = np.zeros((vec_dim,))
        for w in t:
            if w >= 0:
                vec += vocab_emb[w]
            else:
                vec += np.random.random((vec_dim,))
        w_list[index] = vec / len(t)
    return torch.tensor(np.array(w_list)).float()
