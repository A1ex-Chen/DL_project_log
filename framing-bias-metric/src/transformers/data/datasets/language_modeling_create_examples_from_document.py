def create_examples_from_document(self, document: List[List[int]],
    doc_index: int):
    """Creates examples for a single document."""
    max_num_tokens = (self.block_size - self.tokenizer.
        num_special_tokens_to_add(pair=True))
    target_seq_length = max_num_tokens
    if random.random() < self.short_seq_probability:
        target_seq_length = random.randint(2, max_num_tokens)
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                tokens_b = []
                if len(current_chunk) == 1 or random.random(
                    ) < self.nsp_probability:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)
                    for _ in range(10):
                        random_document_index = random.randint(0, len(self.
                            documents) - 1)
                        if random_document_index != doc_index:
                            break
                    random_document = self.documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                input_ids = self.tokenizer.build_inputs_with_special_tokens(
                    tokens_a, tokens_b)
                token_type_ids = (self.tokenizer.
                    create_token_type_ids_from_sequences(tokens_a, tokens_b))
                example = {'input_ids': torch.tensor(input_ids, dtype=torch
                    .long), 'token_type_ids': torch.tensor(token_type_ids,
                    dtype=torch.long), 'next_sentence_label': torch.tensor(
                    1 if is_random_next else 0, dtype=torch.long)}
                self.examples.append(example)
            current_chunk = []
            current_length = 0
        i += 1
