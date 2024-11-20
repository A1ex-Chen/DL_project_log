def get_corpus_from_dataset(self, dataset) ->List[List]:
    image_corpus = []
    for entry in dataset:
        image_corpus.append(entry['image_path'])
    return image_corpus
