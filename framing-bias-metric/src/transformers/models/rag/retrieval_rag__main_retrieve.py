def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int
    ) ->Tuple[np.ndarray, np.ndarray]:
    question_hidden_states_batched = self._chunk_tensor(question_hidden_states,
        self.batch_size)
    ids_batched = []
    vectors_batched = []
    for question_hidden_states in question_hidden_states_batched:
        start_time = time.time()
        ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
        logger.debug('index search time: {} sec, batch size {}'.format(time
            .time() - start_time, question_hidden_states.shape))
        ids_batched.extend(ids)
        vectors_batched.extend(vectors)
    return np.array(ids_batched), np.array(vectors_batched)
