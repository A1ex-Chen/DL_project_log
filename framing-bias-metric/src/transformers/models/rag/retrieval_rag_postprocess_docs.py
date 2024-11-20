def postprocess_docs(self, docs, input_strings, prefix, n_docs,
    return_tensors=None):
    """
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """

    def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        if prefix is None:
            prefix = ''
        out = (prefix + doc_title + self.config.title_sep + doc_text + self
            .config.doc_sep + input_string).replace('  ', ' ')
        return out
    rag_input_strings = [cat_input_and_doc(docs[i]['title'][j], docs[i][
        'text'][j], input_strings[i], prefix) for i in range(len(docs)) for
        j in range(n_docs)]
    contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
        rag_input_strings, max_length=self.config.max_combined_length,
        return_tensors=return_tensors, padding='max_length', truncation=True)
    return contextualized_inputs['input_ids'], contextualized_inputs[
        'attention_mask']
