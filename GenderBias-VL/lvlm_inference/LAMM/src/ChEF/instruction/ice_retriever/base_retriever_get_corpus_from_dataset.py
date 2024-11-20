def get_corpus_from_dataset(self, dataset) ->List[List]:
    corpus = []
    assert 'question' in dataset[0
        ], 'No question in scenarios. You should not use topk_text retriever as the questions in instruction are the same. '
    for entry in dataset:
        corpus.append(entry['question'])
    return corpus
