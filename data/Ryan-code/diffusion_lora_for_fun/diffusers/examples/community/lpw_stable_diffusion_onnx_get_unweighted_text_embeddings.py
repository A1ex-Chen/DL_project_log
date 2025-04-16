def get_unweighted_text_embeddings(pipe, text_input: np.array, chunk_length:
    int, no_boseos_middle: Optional[bool]=True):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            text_input_chunk = text_input[:, i * (chunk_length - 2):(i + 1) *
                (chunk_length - 2) + 2].copy()
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(input_ids=text_input_chunk)[0]
            if no_boseos_middle:
                if i == 0:
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    text_embedding = text_embedding[:, 1:]
                else:
                    text_embedding = text_embedding[:, 1:-1]
            text_embeddings.append(text_embedding)
        text_embeddings = np.concatenate(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(input_ids=text_input)[0]
    return text_embeddings
