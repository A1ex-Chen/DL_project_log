def embed_batch(self, list_inputText, max_length=64, contextual=False):
    """ """
    self.model.eval()
    with torch.no_grad():
        rst = self.embedding_batch(list_inputText[:MAX_BATCH_SENTENCEBERT],
            contextual=contextual)
        itr_additional = int(len(list_inputText) / MAX_BATCH_SENTENCEBERT)
        for i in range(itr_additional):
            start_index = (i + 1) * MAX_BATCH_SENTENCEBERT
            list_candidates = list_inputText[start_index:start_index +
                MAX_BATCH_SENTENCEBERT]
            if len(list_candidates) > 0:
                rst_tmp = self.embedding_batch(list_inputText[start_index:
                    start_index + MAX_BATCH_SENTENCEBERT], contextual=
                    contextual)
                if isinstance(rst_tmp, torch.Tensor):
                    rst = torch.cat((rst, rst_tmp), dim=0)
                else:
                    rst = rst + rst_tmp
        return rst
