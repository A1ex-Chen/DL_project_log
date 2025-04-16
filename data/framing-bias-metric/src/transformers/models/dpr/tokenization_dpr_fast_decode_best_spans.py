def decode_best_spans(self, reader_input: BatchEncoding, reader_output:
    DPRReaderOutput, num_spans: int=16, max_answer_length: int=64,
    num_spans_per_passage: int=4) ->List[DPRSpanPrediction]:
    """
        Get the span predictions for the extractive Q&A model.

        Returns: `List` of `DPRReaderOutput` sorted by descending `(relevance_score, span_score)`. Each
        `DPRReaderOutput` is a `Tuple` with:

            - **span_score**: ``float`` that corresponds to the score given by the reader for this span compared to
              other spans in the same passage. It corresponds to the sum of the start and end logits of the span.
            - **relevance_score**: ``float`` that corresponds to the score of the each passage to answer the question,
              compared to all the other passages. It corresponds to the output of the QA classifier of the DPRReader.
            - **doc_id**: ``int``` the id of the passage.
            - ***start_index**: ``int`` the start index of the span (inclusive).
            - **end_index**: ``int`` the end index of the span (inclusive).

        Examples::

            >>> from transformers import DPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='pt'
            ...     )
            >>> outputs = model(**encoded_inputs)
            >>> predicted_spans = tokenizer.decode_best_spans(encoded_inputs, outputs)
            >>> print(predicted_spans[0].text)  # best span

        """
    input_ids = reader_input['input_ids']
    start_logits, end_logits, relevance_logits = reader_output[:3]
    n_passages = len(relevance_logits)
    sorted_docs = sorted(range(n_passages), reverse=True, key=
        relevance_logits.__getitem__)
    nbest_spans_predictions: List[DPRReaderOutput] = []
    for doc_id in sorted_docs:
        sequence_ids = list(input_ids[doc_id])
        passage_offset = sequence_ids.index(self.sep_token_id, 2) + 1
        if sequence_ids[-1] == self.pad_token_id:
            sequence_len = sequence_ids.index(self.pad_token_id)
        else:
            sequence_len = len(sequence_ids)
        best_spans = self._get_best_spans(start_logits=start_logits[doc_id]
            [passage_offset:sequence_len], end_logits=end_logits[doc_id][
            passage_offset:sequence_len], max_answer_length=
            max_answer_length, top_spans=num_spans_per_passage)
        for start_index, end_index in best_spans:
            start_index += passage_offset
            end_index += passage_offset
            nbest_spans_predictions.append(DPRSpanPrediction(span_score=
                start_logits[doc_id][start_index] + end_logits[doc_id][
                end_index], relevance_score=relevance_logits[doc_id],
                doc_id=doc_id, start_index=start_index, end_index=end_index,
                text=self.decode(sequence_ids[start_index:end_index + 1])))
        if len(nbest_spans_predictions) >= num_spans:
            break
    return nbest_spans_predictions[:num_spans]
