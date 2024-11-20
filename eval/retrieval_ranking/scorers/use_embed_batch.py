def embed_batch(self, list_inputText, max_length=64, contextual=False):
    """ """
    rst = self.model(list_inputText[:MAX_BATCH_USE])
    itr_additional = int(len(list_inputText) / MAX_BATCH_USE)
    for i in range(itr_additional):
        start_index = (i + 1) * MAX_BATCH_USE
        list_candidates = list_inputText[start_index:start_index +
            MAX_BATCH_USE]
        if len(list_candidates) > 0:
            rst_tmp = self.model(list_inputText[start_index:start_index +
                MAX_BATCH_USE])
            rst = np.concatenate((rst, rst_tmp), axis=0)
    return rst
