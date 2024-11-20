def get_doc_dicts(self, doc_ids: np.ndarray) ->List[dict]:
    return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]
