@staticmethod
def _extend_tokens_and_embeddings(tokens, embeddings, tokenizer):
    all_tokens = []
    all_embeddings = []
    for embedding, token in zip(embeddings, tokens):
        if f'{token}_1' in tokenizer.get_vocab():
            multi_vector_tokens = [token]
            i = 1
            while f'{token}_{i}' in tokenizer.added_tokens_encoder:
                multi_vector_tokens.append(f'{token}_{i}')
                i += 1
            raise ValueError(
                f'Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder.'
                )
        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
        if is_multi_vector:
            all_tokens += [token] + [f'{token}_{i}' for i in range(1,
                embedding.shape[0])]
            all_embeddings += [e for e in embedding]
        else:
            all_tokens += [token]
            all_embeddings += [embedding[0]] if len(embedding.shape) > 1 else [
                embedding]
    return all_tokens, all_embeddings
